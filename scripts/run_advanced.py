"""Compare prompt designs and image resolutions on the same validation images."""

from pathlib import Path

from classification import BASELINE, Classifier, advanced_prompts, make_parser, prepare_run, record_setup, write_json
from reporting import compare, copy_test_baseline, evaluate, read_test_baseline, save_comparison_plots


def main():
    parser = make_parser("Compare Qwen3-VL classification and token costs")
    parser.add_argument("--few-shot", type=int, default=3, help="Number of reference images from distinct classes")
    parser.add_argument("--multi-max-new-tokens", type=int, default=96, help="Generation limit for each four-image request")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="After validation, evaluate the baseline and selected variant on the official test split")
    parser.add_argument("--baseline-dir", type=Path,
                        help="Reuse the named baseline test run and evaluate the selected variant on test")
    args = parser.parse_args()
    if args.few_shot < 1:
        parser.error("--few-shot must be positive")
    if args.multi_max_new_tokens <= 0:
        parser.error("--multi-max-new-tokens must be positive")
    output, dataset = prepare_run(parser, args, "advanced")
    try:
        prompts = advanced_prompts(dataset, args.seed, args.few_shot)
    except ValueError as exc:
        parser.error(str(exc))
    reused_baseline = None
    if args.baseline_dir is not None:
        try:
            reused_baseline = read_test_baseline(args.baseline_dir, args, dataset, prompts[0])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))

    classifier = Classifier(args.model_dir, args.device, args.max_new_tokens, args.gpu_memory_fraction)
    record_setup(output, args, dataset, classifier, prompts)

    def evaluate_prompt(split_output, samples, prompt):
        limit = args.multi_max_new_tokens if prompt.targets_per_request > 1 else args.max_new_tokens
        return evaluate(split_output / prompt.name, samples, dataset.labels, classifier, prompt,
                        seed=args.seed, max_new_tokens=limit)

    validation = output / "validation"
    results = {
        prompt.name: evaluate_prompt(validation, dataset.validation, prompt)
        for prompt in prompts
    }
    comparisons = compare(validation, results)
    save_comparison_plots(validation, comparisons)

    selected = min(comparisons, key=lambda row: (
        -row["correct"], row["mean_total_tokens"], row["configuration"],
    ))
    name = selected["configuration"]
    write_json(output / "selection.json", {
        "split": "validation",
        "rule": "Most correct images, then lowest mean total tokens, then configuration name",
        "selected": selected,
        "pareto_configurations": [row["configuration"] for row in comparisons if row["pareto_optimal"]],
    })
    if args.evaluate_test or reused_baseline is not None:
        test = output / "test"
        if reused_baseline is not None:
            copy_test_baseline(args.baseline_dir, test / BASELINE)
            test_results = {BASELINE: reused_baseline}
        else:
            test_results = {BASELINE: evaluate_prompt(test, dataset.test, prompts[0])}
        if name != BASELINE:
            selected_prompt = next(prompt for prompt in prompts if prompt.name == name)
            test_results[name] = evaluate_prompt(test, dataset.test, selected_prompt)
        compare(test, test_results)

    print(f"Selected on validation: {name} ({selected['accuracy_pct']:.2f}%, "
          f"{selected['mean_total_tokens']:.1f} mean total tokens)")
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
