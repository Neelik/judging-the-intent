import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from peewee import JOIN
from judging_the_intent.db.schema import (
    Intent,
    Dataset,
    Query
)
from tqdm import tqdm
import logging
LOGGER = logging.getLogger(__name__)


def clean_intents(dataset_identifier: str):
    dataset_queries = (
        Query.select()
        .where(Query.dataset_name_id.in_(Dataset.select().where(Dataset.name == dataset_identifier)))
        .alias("dataset_queries")
    )

    intents_to_update = (
        Intent.select(
            Intent,
            Query.id.alias("query_id")
        )
        .join(Query, JOIN.LEFT_OUTER, on=Intent.query_id == Query.id)
        .where((Intent.source == "generated") & (Intent.query_id.in_(dataset_queries)))
    )

    intents_to_update_tuples = intents_to_update.namedtuples()
    for intent_to_update in tqdm(intents_to_update_tuples, total=intents_to_update.count(), desc=">> Cleaning intent text..."):
        updated_text = intent_to_update.text.replace("_", " ").replace("<", "").replace(">", "")
        updated_text_no_nums = re.sub(r"\d+", "", updated_text)
        updated_text_no_punc = re.sub(r'[^a-zA-Z0-9\s]', '', updated_text_no_nums)
        updated_text = updated_text_no_punc.strip()

        # Remove clueweb variations
        updated_text = re.sub(r"(?i)clueweben", "", updated_text)
        updated_text = re.sub(r"(?i)clueweb", "", updated_text)

        final_text = updated_text.strip().lower()

        update_query = (
            Intent.update(text=final_text)
            .where(Intent.id == intent_to_update.id)
        )
        update_query.execute()

def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=False, help="IR Datasets dataset identifiers.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='{levelname} - {asctime} - {module} - {message}', style="{",
                        datefmt="%Y-%m-%d %H:%M")

    for dataset in args.datasets:
        clean_intents(dataset)

if __name__ == "__main__":
    main()