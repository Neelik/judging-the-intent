import logging
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from judging_the_intent.util.eval import Evaluator

LOGGER = logging.getLogger(__file__)


def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--model", required=True, help="HuggingFace model identifiers."
    )
    ap.add_argument("--datasets", dest="datasets", required=True, nargs="+", help="Dataset identifiers")
    ap.add_argument("--qrels_true_path", help="Directory containing the TREC Qrels files.")
    ap.add_argument("--intent_aware", action="store_true", default=False, help="Run the intent-aware evaluation.")
    ap.add_argument("--checkpointed_model", action="store_true", help="Set to true if evaluation is on annotations from a fine-tuned model.")
    ap.add_argument("--prompt_style", required=True, choices=("human", "human-intent", "binary", "binary-intent", "dna", "dna-intent"), help="Prompt style identifier.")
    ap.add_argument("--intent_source", type=str, choices=("human", "generated"), help="Intent source identifier.")
    args = ap.parse_args()

    # sanity check
    if "intent" in args.prompt_style and not args.intent_aware:
        LOGGER.warning("Intent aware evaluation requires --intent_aware.")
        sys.exit(1)

    checkpointed = False
    if args.checkpointed_model:
        checkpointed = True

    logging.basicConfig(level=logging.INFO)

    for dataset in args.datasets:
        LOGGER.info(f"Evaluating {args.model} annotations of {dataset}.")
        Evaluator(model=args.model, dataset=dataset, target_type="binary", prompt_style=args.prompt_style,
                  intent_aware=args.intent_aware, checkpointed_model=checkpointed,
                  qrels_true_path=args.qrels_true_path, intent_source=args.intent_source).evaluate()

if __name__ == "__main__":
    main()