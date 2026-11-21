"""Run the zero-shot image classification baseline."""

from classification import Classifier, baseline_prompt, make_parser, prepare_run, record_setup
from reporting import evaluate, save_baseline_plots


def main():
    parser = make_parser("Run the zero-shot image classification baseline")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    args = parser.parse_args()
    output, dataset = prepare_run(parser, args, "baseline")
    samples = getattr(dataset, args.split)
    prompt = baseline_prompt(dataset.labels)
    classifier = Classifier(args.model_dir, args.device, args.max_new_tokens, args.gpu_memory_fraction)
    record_setup(output, args, dataset, classifier, [prompt])
    metrics, rows = evaluate(output, samples, dataset.labels, classifier, prompt,
                             seed=args.seed, max_new_tokens=args.max_new_tokens)
    save_baseline_plots(output, dataset.labels, rows)
    print(f"{args.split}: {metrics['correct']}/{metrics['images']} ({metrics['accuracy_pct']:.2f}%)")
    print(f"Mean total tokens: {metrics['mean_total_tokens']:.1f}\nResults: {output.resolve()}")


if __name__ == "__main__":
    main()
