import logging
from judging_the_intent.db import DATABASE
from judging_the_intent.db.schema import (
    Annotation,
    Config,
    Dataset,
    Document,
    Intent,
    Query,
    Triple,
)

LOGGER = logging.getLogger(__file__)


def main():
    logging.basicConfig(level=logging.INFO)

    with DATABASE:
        LOGGER.info("\tCreating initial tables...")
        DATABASE.create_tables([Dataset, Query, Intent, Document, Triple, Config, Annotation])

    # Test that the tables exist
    with DATABASE:
        tables = DATABASE.get_tables()
        LOGGER.info(f"\t{[t for t in tables]} successfully created.")

if __name__ == "__main__":
    main()