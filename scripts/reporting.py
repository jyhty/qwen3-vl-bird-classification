"""Persist predictions, compute experiment metrics, and render report figures."""

from collections import Counter
import csv
from itertools import batched
import json
from math import ceil
import random
import shutil
from statistics import mean, median

from classification import AnswerParser, BASELINE, EVALUATION_VERSION, write_json

REQUEST_TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")
UNPARSED = "[unparsed]"
DISPLAY_NAMES = {
    BASELINE: "Baseline",
    "roleplay": "Professional classifier",
    "few_shot": "Few-shot",
    "resolution_96": "96 visual tokens",
    "multi_image": "Four target images",
}
COLORS = {"blue": "#23728B", "orange": "#D6732C", "grey": "#8794A1"}


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(output, samples, labels, classifier, prompt, *, seed, max_new_tokens):
    from tqdm import tqdm

    output.mkdir(parents=True, exist_ok=True)
    parser = AnswerParser(labels)
    multi = prompt.targets_per_request > 1
    ordered = list(samples)
    if multi:
        random.Random(seed).shuffle(ordered)
    groups = list(batched(ordered, prompt.targets_per_request))
    predictions, requests = [], []
    started = classifier.start_group()
    predict = classifier.prediction_function(prompt, max_new_tokens)
    with (
        (output / "predictions.jsonl").open("w", encoding="utf-8", buffering=1) as file,
        (output / "requests.jsonl").open("w", encoding="utf-8", buffering=1) as request_file,
    ):
        for request_id, group in enumerate(tqdm(groups, desc=prompt.name, unit="request"), 1):
            result = predict(group)
            parts = result.pop("targets")
            raw = result["raw_output"]
            if multi:
                answers, diagnostics = parser.parse_numbered(raw, len(group))
            else:
                answers, diagnostics = [(raw, *parser.parse(raw))], {}
            shared_input = (result["input_tokens"] - sum(part["image_tokens"] for part in parts)) / len(group)
            output_tokens = result["output_tokens"] / len(group)
            request = {
                "request_id": request_id,
                "image_ids": [sample.image_id for sample in group],
                **result,
                **diagnostics,
            }
            request_file.write(json.dumps(request, ensure_ascii=False) + "\n")
            requests.append(request)

            for number, (sample, part, answer) in enumerate(zip(group, parts, answers, strict=True), 1):
                excerpt, label, status = answer
                input_tokens = part["image_tokens"] + shared_input
                row = {
                    "image_id": sample.image_id,
                    "request_id": request_id,
                    "image_number": number,
                    "true_label": sample.label,
                    "predicted_label": label,
                    "parse_status": status,
                    "correct": label == sample.label,
                    "raw_output": raw,
                    "answer_text": excerpt,
                    **part,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "few_shot_image_tokens": result["few_shot_image_tokens"] / len(group),
                    "hit_generation_limit": result["hit_generation_limit"],
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                predictions.append(row)
    profile = classifier.finish_group(started)
    return write_metrics(output, labels, predictions, requests, profile, max_new_tokens), predictions


def write_metrics(output, labels, rows, requests, profile, max_new_tokens):
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    counts = Counter((row["true_label"], row["predicted_label"]) for row in rows)
    statuses = Counter(row["parse_status"] for row in rows)
    values = sorted(row["total_tokens"] for row in rows)
    total_tokens = {field: sum(req[field] for req in requests) for field in REQUEST_TOKEN_FIELDS}
    metrics = {
        "images": total,
        "requests": len(requests),
        "correct": correct,
        "accuracy_pct": 100 * correct / total,
        "unparsed": sum(row["predicted_label"] is None for row in rows),
        "parse_status_counts": dict(statuses),
        "hit_generation_limit": sum(req["hit_generation_limit"] for req in requests),
        "max_new_tokens": max_new_tokens,
        "invalid_numbers": sum(len(req.get("invalid_numbers", [])) for req in requests),
        "unnumbered_lines": sum(len(req.get("unnumbered_lines", [])) for req in requests),
        "missing_numbers": statuses["missing_number"],
        "duplicate_numbers": statuses["duplicate_number"],
        **{f"mean_{field}": count / total for field, count in total_tokens.items()},
        "token_sums": total_tokens,
        "mean_image_tokens": mean(row["image_tokens"] for row in rows),
        "mean_few_shot_image_tokens": sum(req["few_shot_image_tokens"] for req in requests) / total,
        "median_total_tokens": median(values),
        "p90_total_tokens": values[ceil(0.9 * total) - 1],
        **profile,
        "mean_request_seconds": mean(req["seconds"] for req in requests),
        "seconds_per_image": profile["elapsed_seconds"] / total,
    }
    write_json(output / "metrics.json", metrics)

    per_class, confusion = [], []
    for true in labels:
        observed = sum(counts[true, pred] for pred in (*labels, None))
        hits = counts[true, true]
        per_class.append({
            "label": true, "images": observed, "correct": hits,
            "accuracy_pct": 100 * hits / observed if observed else None,
        })
        confusion.append({
            "true_label": true,
            **{pred: counts[true, pred] for pred in labels},
            UNPARSED: counts[true, None],
        })
    write_csv(output / "per_class.csv", per_class)
    write_csv(output / "confusion_matrix.csv", confusion)
    return metrics


def read_test_baseline(source, args, dataset, prompt):
    with (source / "run_config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    expected = {
        "split": "test",
        "data_dir": str(args.data_dir.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
    }
    if (config.get("evaluation_version") != EVALUATION_VERSION
            or config.get("prompts") != [prompt.metadata()]
            or any(config["arguments"].get(key) != value for key, value in expected.items())):
        raise ValueError(f"Incompatible test baseline in {source}; use the new baseline script with matching settings")
    with (source / "predictions.jsonl").open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file]
    with (source / "metrics.json").open(encoding="utf-8") as file:
        metrics = json.load(file)
    if ({(row["image_id"], row["true_label"]) for row in rows}
            != {(sample.image_id, sample.label) for sample in dataset.test}
            or not (len(rows) == len(dataset.test) == metrics["images"] == metrics["requests"])
            or metrics["correct"] != sum(row["correct"] for row in rows)):
        raise ValueError(f"Test baseline in {source} is incomplete or uses different test images")
    for name in ("requests.jsonl", "per_class.csv", "confusion_matrix.csv"):
        if not (source / name).is_file():
            raise ValueError(f"Missing baseline artifact: {source / name}")
    return {**metrics, "reused": True}, rows


def copy_test_baseline(source, destination):
    destination.mkdir(parents=True)
    for name in ("run_config.json", "predictions.jsonl", "metrics.json", "requests.jsonl",
                 "per_class.csv", "confusion_matrix.csv"):
        shutil.copy2(source / name, destination / name)
    write_json(destination / "reused_from.json", {"directory": str(source.resolve())})


def save_baseline_plots(output, labels, rows):
    plt = pyplot()
    totals = Counter(row["true_label"] for row in rows)
    hits = Counter(row["true_label"] for row in rows if row["correct"])
    ordered = sorted(labels, key=lambda label: (hits[label] / totals[label], label))
    rates = [100 * hits[label] / totals[label] for label in ordered]
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(labels) + 1.7), layout="constrained")
    bars = ax.barh(ordered, rates, color=COLORS["blue"])
    ax.bar_label(bars, labels=[f"{hits[label]}/{totals[label]}" for label in ordered], padding=4)
    ax.set(xlim=(0, 113), xticks=range(0, 101, 20),
           xlabel="Accuracy (%)", title="Baseline accuracy by class")
    ax.xaxis.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.savefig(output / "per_class_accuracy.png", dpi=200)
    plt.close(fig)

    predicted = (*labels, None)
    counts = Counter((row["true_label"], row["predicted_label"]) for row in rows)
    matrix = [[counts[true, pred] for pred in predicted] for true in labels]
    fig, ax = plt.subplots(figsize=(13, 11), layout="constrained")
    heatmap = ax.imshow(matrix, cmap="Blues", vmin=0, interpolation="nearest")
    largest = max(max(row) for row in matrix)
    for i, row in enumerate(matrix):
        for j, count in enumerate(row):
            if count:
                ax.text(j, i, str(count), ha="center", va="center", fontsize=8,
                        color="white" if count > largest / 2 else "#163140")
    ax.set_xticks(range(len(predicted)), [*labels, UNPARSED], rotation=55, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted class", ylabel="True class", title="Baseline confusion matrix")
    fig.colorbar(heatmap, ax=ax, shrink=0.72, label="Images")
    fig.savefig(output / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def compare(output, results):
    baseline_metrics, baseline_rows = results[BASELINE]
    comparisons = []
    with (output / "paired_changes.jsonl").open("w", encoding="utf-8") as file:
        for name, (metrics, rows) in results.items():
            improved = regressed = 0
            by_id = {row["image_id"]: row for row in rows}
            if len(rows) != len(by_id) or len(rows) != len(baseline_rows):
                raise ValueError(f"Unaligned evaluation samples for {name}")
            for base in baseline_rows:
                current = by_id.get(base["image_id"])
                if current is None or base["true_label"] != current["true_label"]:
                    raise ValueError(f"Unaligned evaluation samples for {name}")
                if base["correct"] == current["correct"]:
                    continue
                improved += current["correct"]
                regressed += base["correct"]
                file.write(json.dumps({
                    "configuration": name,
                    "image_id": current["image_id"],
                    "true_label": current["true_label"],
                    "change": "wrong_to_right" if current["correct"] else "right_to_wrong",
                    "baseline": {key: base[key] for key in (
                        "predicted_label", "parse_status", "raw_output", "answer_text", "total_tokens",
                    )},
                    "variant": {key: current[key] for key in (
                        "predicted_label", "parse_status", "raw_output", "answer_text", "total_tokens",
                        "request_id", "image_number",
                    )},
                }, ensure_ascii=False) + "\n")
            comparisons.append({
                "configuration": name,
                "reused": metrics.get("reused", False),
                **{key: metrics[key] for key in (
                    "images", "requests", "correct", "accuracy_pct", "unparsed",
                    "mean_input_tokens", "mean_output_tokens", "mean_total_tokens",
                    "mean_image_tokens", "mean_few_shot_image_tokens", "p90_total_tokens",
                    "invalid_numbers", "unnumbered_lines", "missing_numbers",
                    "duplicate_numbers", "hit_generation_limit",
                    "elapsed_seconds", "mean_request_seconds", "seconds_per_image",
                    "peak_allocated_bytes", "peak_reserved_bytes",
                )},
                "accuracy_delta_pp": metrics["accuracy_pct"] - baseline_metrics["accuracy_pct"],
                "mean_total_tokens_delta": metrics["mean_total_tokens"] - baseline_metrics["mean_total_tokens"],
                "wrong_to_right": improved,
                "right_to_wrong": regressed,
            })

    for row in comparisons:
        accuracy, cost = row["accuracy_pct"], row["mean_total_tokens"]
        row["pareto_optimal"] = not any(
            other["accuracy_pct"] >= accuracy and other["mean_total_tokens"] <= cost
            and (other["accuracy_pct"] > accuracy or other["mean_total_tokens"] < cost)
            for other in comparisons
        )
    write_csv(output / "comparison.csv", comparisons)
    return comparisons


def save_comparison_plots(output, comparisons):
    plt = pyplot()
    labels = [DISPLAY_NAMES[row["configuration"]] for row in comparisons]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    # Numbered points keep nearby accuracy/cost values readable.
    for i, (row, label) in enumerate(zip(comparisons, labels, strict=True), 1):
        color = COLORS["blue"] if row["pareto_optimal"] else COLORS["grey"]
        x, y = row["mean_total_tokens"], row["accuracy_pct"]
        ax.scatter(x, y, color=color, s=60, label=f"{i}. {label}")
        ax.annotate(str(i), (x, y), xytext=(4, (6, -11, 14)[(i - 1) % 3]),
                    textcoords="offset points", fontsize=9)
    ax.margins(x=0.08, y=0.15)
    ax.set(xlabel="Mean total tokens per evaluated image", ylabel="Accuracy (%)",
           title="Accuracy vs. token cost (blue = Pareto-optimal)")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    fig.savefig(output / "accuracy_vs_tokens.png", dpi=200)
    plt.close(fig)

    inputs = [row["mean_input_tokens"] for row in comparisons]
    outputs = [row["mean_output_tokens"] for row in comparisons]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.bar(labels, inputs, color=COLORS["blue"], label="Input")
    bars = ax.bar(labels, outputs, bottom=inputs, color=COLORS["orange"], label="Output")
    ax.bar_label(bars, labels=[f"{i + o:.1f}" for i, o in zip(inputs, outputs, strict=True)], padding=4)
    ax.tick_params(axis="x", labelrotation=22)
    ax.margins(y=0.15)
    ax.set(ylabel="Mean tokens per evaluated image", title="Input and output token cost")
    ax.legend(frameon=False)
    fig.savefig(output / "token_breakdown.png", dpi=200)
    plt.close(fig)
