import logging
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from judging_the_intent.util.generator import IntentGenerator
from judging_the_intent.util.prompter import IntentGenerationPrompter, SubtopicGenerationPrompter
LOGGER = logging.getLogger(__name__)


def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--model", required=True, default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model identifier. May require configuration of HF access token."
    )
    ap.add_argument("--datasets", nargs="+", required=False, help="IR Datasets dataset identifiers.")
    ap.add_argument("--prompt_style", required=True, choices=("intent", "subtopic"), type=str, help="Define the prompt style to use in this intent generation run.")
    ap.add_argument("--db_commit", action="store_true", default=False, help="Write the generated Intents/Subtopics to database.")
    ap.add_argument("--save_to_disk", action="store_true", default=False, help="Save the generated Intents/Subtopics to disk.")
    args = ap.parse_args()

    if args.prompt_style == "intent":
        prompter = IntentGenerationPrompter(prompt_style="generate-intent")
    elif args.prompt_style == "subtopic":
        prompter = SubtopicGenerationPrompter(prompt_style="generate-subtopic")
    else:
        # Setting this to clean up the warning, but realistically this code will never run due to choices argument in the parser
        prompter = None

    for dataset in args.datasets:
        LOGGER.info(f"\tGenerating intents for {dataset}...")
        generator = IntentGenerator(model=args.model, dataset=dataset, prompter=prompter)
        generator.generate(commit_to_db=args.db_commit, write_to_file=args.save_to_disk)

if __name__ == "__main__":
    main()