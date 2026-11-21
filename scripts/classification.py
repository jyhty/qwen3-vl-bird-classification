"""Data splits, prompt construction, and local Qwen3-VL prediction."""

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from importlib.metadata import version
import json
from math import prod
from pathlib import Path
import random
import re
from time import perf_counter
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
QUESTION = "What species of bird is shown in this image?"
BASELINE = "baseline"
EVALUATION_VERSION = 3
ROLEPLAY = "You are a professional image classifier."
TARGETS_PER_MULTI_REQUEST = 4

ALIASES = {
    "arctic tern": "artic tern",
    "bairds sparrow": "baird sparrow",
    "scotts oriole": "scott oriole",
    "roadrunner": "geococcyx",
}


@dataclass(frozen=True)
class Sample:
    path: Path
    image_id: str
    label: str


@dataclass(frozen=True)
class Dataset:
    labels: tuple[str, ...]
    reference: tuple[Sample, ...]
    validation: tuple[Sample, ...]
    test: tuple[Sample, ...]


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str
    examples: tuple[Sample, ...] = ()
    max_image_tokens: int | None = None
    targets_per_request: int = 1

    def metadata(self):
        return {
            "name": self.name,
            "prompt": self.text,
            "few_shot_examples": [
                {"image_id": sample.image_id, "answer": sample.label}
                for sample in self.examples
            ],
            "max_visual_tokens_per_image": self.max_image_tokens,
            "targets_per_request": self.targets_per_request,
        }


def load_dataset(root, seed, validation_fraction):
    def load_split(split):
        groups = defaultdict(list)
        for path in sorted((root / split).glob("*/*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                sample = Sample(path, path.relative_to(root).as_posix(), path.parent.name)
                groups[sample.label].append(sample)
        if not groups:
            raise ValueError(f"No class-folder images found in {root / split}")
        return groups

    train = load_split("train")
    test = load_split("test")
    if set(train) != set(test):
        raise ValueError("Train and test must have the same class names")

    labels = tuple(sorted(train))
    rng = random.Random(seed)
    reference, validation = [], []
    for label in labels:
        samples = train[label]
        if len(samples) < 2:
            raise ValueError(f"Class {label!r} needs at least two training images")
        count = min(len(samples) - 1, max(1, round(len(samples) * validation_fraction)))
        held_out = set(rng.sample(samples, count))
        validation.extend(sample for sample in samples if sample in held_out)
        reference.extend(sample for sample in samples if sample not in held_out)

    return Dataset(labels, tuple(reference), tuple(validation),
                   tuple(sample for label in labels for sample in test[label]))


def candidate_context(labels):
    return "This image may belong to one of the following categories: " + ", ".join(labels) + "."


def baseline_prompt(labels):
    return Prompt(BASELINE, f"{candidate_context(labels)}\n{QUESTION}\nReply with exactly one category label.")


def advanced_prompts(dataset, seed, shots):
    base = baseline_prompt(dataset.labels)
    groups = defaultdict(list)
    for sample in dataset.reference:
        groups[sample.label].append(sample)
    if shots > len(groups):
        raise ValueError(f"Few-shot example count must be at most {len(groups)} (one per class)")
    rng = random.Random(seed)
    examples = tuple(rng.choice(groups[label]) for label in rng.sample(dataset.labels, shots))
    multiple = (
        "Classify each numbered image. Candidate categories: " + ", ".join(dataset.labels) + ".\n"
        "Return one line per image, using the format `1: category label`, `2: category label`, and so on. "
        "Use only the candidate categories."
    )
    return (
        base,
        replace(base, name="roleplay", text=f"{ROLEPLAY} {base.text}"),
        replace(base, name="few_shot", examples=examples),
        replace(base, name="resolution_96", max_image_tokens=96),
        Prompt("multi_image", multiple, targets_per_request=TARGETS_PER_MULTI_REQUEST),
    )


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "").replace("'", "")
    return " ".join(re.findall(r"[^\W_]+", text))


class AnswerParser:
    def __init__(self, labels):
        self.labels = {normalize(label): label for label in labels}
        self.labels.update({alias: label for alias, label in ALIASES.items() if label in labels})
        terms = sorted(self.labels, key=len, reverse=True)
        self.pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, terms)) + r")(?!\w)")

    def parse(self, text):
        normalized = normalize(text)
        if normalized in self.labels:
            return self.labels[normalized], "exact"
        labels = {self.labels[match.group()] for match in self.pattern.finditer(normalized)}
        if len(labels) == 1:
            return labels.pop(), "unique_mention"
        return None, "ambiguous" if labels else "unmatched"

    def parse_numbered(self, text, count):
        answers = defaultdict(list)
        invalid_numbers, unnumbered_lines = [], []
        for line in unicodedata.normalize("NFKC", text).splitlines():
            if not line.strip():
                continue
            match = re.fullmatch(r"\s*(\d{1,6})\s*[:.)]\s*(.*?)\s*", line)
            if not match:
                unnumbered_lines.append(line)
                continue
            number = int(match[1])
            if 1 <= number <= count:
                answers[number].append(match[2])
            else:
                invalid_numbers.append(number)

        parsed = []
        for number in range(1, count + 1):
            entries = answers[number]
            if len(entries) == 1:
                parsed.append((entries[0], *self.parse(entries[0])))
            else:
                parsed.append(("\n".join(entries), None, "duplicate_number" if entries else "missing_number"))
        return parsed, {"invalid_numbers": invalid_numbers, "unnumbered_lines": unnumbered_lines}


class Classifier:
    def __init__(self, model_dir, device, max_new_tokens, gpu_memory_fraction=None):
        started = perf_counter()
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.device = torch.device(device)
        self.gpu_memory_fraction = gpu_memory_fraction
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Use the GPU session or pass --device cpu.")
            with torch.cuda.device(self.device):
                bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
                if gpu_memory_fraction is not None:
                    torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, self.device)
            self.dtype = torch.bfloat16 if bf16 else torch.float16
        else:
            self.dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, dtype=self.dtype,
            device_map={"": self.device}, attn_implementation="sdpa",
        ).eval()
        self.generation = deepcopy(self.model.generation_config)
        self.generation.do_sample = False
        self.generation.temperature = None
        self.generation.top_p = None
        self.generation.top_k = None
        self.generation.max_new_tokens = max_new_tokens
        self.generation.use_cache = True
        eos = self.generation.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.patch_size = self.processor.image_processor.patch_size
        self.merge_size = self.processor.image_processor.merge_size
        self.synchronize()
        self.load_seconds = perf_counter() - started

    @staticmethod
    def open_image(sample):
        from PIL import Image

        with Image.open(sample.path) as image:
            return image.convert("RGB")

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def start_group(self):
        if self.device.type == "cuda":
            self.synchronize()
            with self.torch.cuda.device(self.device):
                self.torch.cuda.empty_cache()
            self.torch.cuda.reset_peak_memory_stats(self.device)
        return perf_counter()

    def finish_group(self, started):
        self.synchronize()
        cuda = self.torch.cuda
        is_cuda = self.device.type == "cuda"
        return {
            "elapsed_seconds": perf_counter() - started,
            "peak_allocated_bytes": cuda.max_memory_allocated(self.device) if is_cuda else None,
            "peak_reserved_bytes": cuda.max_memory_reserved(self.device) if is_cuda else None,
        }

    def metadata(self):
        return {
            "device": str(self.device),
            "gpu_name": self.torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "dtype": str(self.dtype),
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "max_new_tokens": self.generation.max_new_tokens,
            "do_sample": False,
            "output_token_count_includes_eos": True,
            "attention": "sdpa",
            "use_cache": True,
        }

    def prediction_function(self, prompt, max_new_tokens):
        pictures = [self.open_image(sample) for sample in prompt.examples]
        messages = []
        for sample, picture in zip(prompt.examples, pictures, strict=True):
            messages.extend([
                {"role": "user", "content": [
                    {"type": "image", "image": picture},
                    {"type": "text", "text": prompt.text},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": sample.label}]},
            ])

        def predict(samples):
            self.synchronize()
            started = perf_counter()
            targets = [self.open_image(sample) for sample in samples]
            content = []
            for number, picture in enumerate(targets, 1):
                if prompt.targets_per_request > 1:
                    content.append({"type": "text", "text": f"Image {number}:"})
                content.append({"type": "image", "image": picture})
            content.append({"type": "text", "text": prompt.text})
            query = {"role": "user", "content": content}
            text = self.processor.apply_chat_template(
                [*messages, query], tokenize=False, add_generation_prompt=True,
            )
            options = {}
            if prompt.max_image_tokens is not None:
                pixels_per_visual_token = (self.patch_size * self.merge_size) ** 2
                options["size"] = {
                    **self.processor.image_processor.size,
                    "shortest_edge": pixels_per_visual_token,
                    "longest_edge": prompt.max_image_tokens * pixels_per_visual_token,
                }
            inputs = self.processor(
                text=[text], images=[*pictures, *targets], return_tensors="pt", **options,
            )
            grids = inputs["image_grid_thw"].tolist()
            if len(grids) != len(pictures) + len(targets):
                raise RuntimeError("Unexpected image grid count for the request")
            visual_tokens = [prod(grid) // self.merge_size**2 for grid in grids]
            input_tokens = int(inputs["attention_mask"].sum().item())
            prompt_length = inputs["input_ids"].shape[1]
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **inputs.to(self.device), generation_config=self.generation,
                    use_model_defaults=False, max_new_tokens=max_new_tokens,
                )
            generated_ids = generated[0, prompt_length:].tolist()
            response = self.processor.decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            self.synchronize()
            return {
                "raw_output": response,
                "input_tokens": input_tokens,
                "output_tokens": len(generated_ids),
                "total_tokens": input_tokens + len(generated_ids),
                "few_shot_image_tokens": sum(visual_tokens[:len(pictures)]),
                "targets": [
                    {
                        "original_size": list(picture.size),
                        "processed_size": [grid[2] * self.patch_size, grid[1] * self.patch_size],
                        "image_tokens": tokens,
                    }
                    for picture, grid, tokens in zip(
                        targets, grids[len(pictures):], visual_tokens[len(pictures):], strict=True,
                    )
                ],
                "hit_generation_limit": len(generated_ids) >= max_new_tokens
                and generated_ids[-1] not in self.eos_ids,
                "seconds": perf_counter() - started,
            }

        return predict


def make_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/simple_classification")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output-dir", type=Path, help="Use an empty directory for a new run")
    parser.add_argument("--device", default="cuda:0", help="For example, cuda:0 or cpu")
    parser.add_argument("--gpu-memory-fraction", type=float,
                        help="Cap this process's PyTorch CUDA allocator at this fraction of GPU memory")
    parser.add_argument("--seed", type=int, default=42, help="Validation split and few-shot seed")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    return parser


def prepare_run(parser, args, name):
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if not re.fullmatch(r"cpu|cuda(?::\d+)?", args.device):
        parser.error("--device must be cpu, cuda, or cuda:N")
    if args.gpu_memory_fraction is not None:
        if not 0 < args.gpu_memory_fraction <= 1:
            parser.error("--gpu-memory-fraction must be greater than 0 and at most 1")
        if args.device == "cpu":
            parser.error("--gpu-memory-fraction requires a CUDA device")
    if not (args.model_dir / "config.json").is_file():
        parser.error(f"No local model config found in {args.model_dir}")
    try:
        dataset = load_dataset(args.data_dir, args.seed, args.validation_fraction)
    except ValueError as exc:
        parser.error(str(exc))

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    output = args.output_dir or ROOT / "outputs" / f"{name}-{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    return output, dataset


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def record_setup(output, args, dataset, classifier, prompts):
    write_json(output / "run_config.json", {
        "evaluation_version": EVALUATION_VERSION,
        "arguments": {key: str(value.resolve()) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "output_dir": str(output.resolve()),
        "model": classifier.metadata(),
        "model_load_seconds": classifier.load_seconds,
        "versions": {name: version(name) for name in ("torch", "transformers", "Pillow", "matplotlib")},
        "labels": dataset.labels,
        "label_aliases": {alias: label for alias, label in ALIASES.items() if label in dataset.labels},
        "prompts": [prompt.metadata() for prompt in prompts],
    })
    write_json(output / "split.json", {
        name: [sample.image_id for sample in getattr(dataset, name)]
        for name in ("reference", "validation", "test")
    })
