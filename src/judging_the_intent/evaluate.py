import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from judging_the_intent.util.eval import Evaluator

LOGGER = logging.getLogger(__file__)


def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--model", required=True, help="HuggingFace model identifiers."
    )
    ap.add_argument("--datasets", dest="datasets", required=True, nargs="+", help="Dataset identifiers")
    ap.add_argument("--qrels_true_path", help="Directory containing the TREC Qrels files.",
                    default=str(Path(__file__).parent.parent.parent.joinpath("trec-web", "qrels")))
    ap.add_argument("--intent_aware", action="store_true", default=False, help="Run the intent-aware evaluation.")
    ap.add_argument("--checkpointed_model", action="store_true", help="Set to true if evaluation is on annotations from a fine-tuned model.")
    args = ap.parse_args()

    checkpointed = False
    if args.checkpointed_model:
        checkpointed = True

    logging.basicConfig(level=logging.INFO)

    for dataset in args.datasets:
        LOGGER.info(f"Evaluating {args.model} annotations of {dataset}.")
        Evaluator(args.model, args.qrels_true_path, dataset, "binary", args.intent_aware, checkpointed).evaluate()

if __name__ == "__main__":
    main()