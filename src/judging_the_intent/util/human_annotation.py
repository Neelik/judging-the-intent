import csv
import ir_datasets
from ir_datasets_subsample import register_subsamples
from collections import namedtuple
from pathlib import Path

from peewee import IntegrityError
from tqdm import tqdm
from judging_the_intent import __version__
from judging_the_intent.db.schema import (
    Annotation,
    Config,
    Dataset,
    Query,
    Document,
    Intent,
    Triple,
)
import logging
LOGGER = logging.getLogger(__name__)


def load_qrels_as_human_annotations(dataset_identifier, intent_aware=False):
    """
        Function that will load QRels from ir_datasets into the database as annotations from a human model with a human
        prompt style. Useful when wanting to make evaluation entirely database driven.

        :param dataset_identifier: Name of the dataset from which to load QRels
        :param intent_aware: Flag to identify whether to include Intent in Triple retrieval

    """
    config, created = Config.get_or_create(
        model_name="human", version=__version__, with_intent=False,
        fine_tuned=False, prompt_style="human" if not intent_aware else "human-intent",
    )
    if created:
        LOGGER.info(
            "model %s (version %s) not found in DB, creating",
            "human",
            __version__,
        )
    else:
        LOGGER.info("found model %s (version %s) in DB", "human", __version__)

    dataset_db = Dataset.select().where(Dataset.name == dataset_identifier)
    document_collections = []

    # Early detection and branching to handle the case of DL-MIA, which is based on two underlying collections
    if "dl-mia" in dataset_identifier:
        if intent_aware:
            DLMIAQrel = namedtuple("DLMIAQrel", ['query_id', 'intent_id', 'doc_id', 'relevance'])
            data_path = Path(__file__).parent.parent.parent.parent.joinpath("datasets", dataset_identifier)
            qrels_path = Path(data_path).joinpath("qrels", f"{dataset_identifier}.qid_iid_qrel.txt")

            original_qrels = []
            with open(qrels_path, "r") as qrels_file:
                for q_id, i_id, d_id, rel in csv.reader(qrels_file, delimiter=" "):
                    original_qrels.append(DLMIAQrel(q_id, i_id, d_id, rel))

            document_collections.append(original_qrels)

        else:
            dl_2021 = ir_datasets.load("msmarco-passage-v2/trec-dl-2021")
            dl_2022 = ir_datasets.load("msmarco-passage-v2/trec-dl-2022")
            document_collections.append(dl_2021)
            document_collections.append(dl_2022)

    elif "clueweb" in dataset_identifier:
        register_subsamples()

        dataset_name_split = dataset_identifier.split("/")
        dataset_top_level_name = dataset_name_split[1]
        dataset_track = dataset_name_split[-1]
        data_path = Path(__file__).parent.parent.parent.parent.joinpath("datasets", dataset_top_level_name, dataset_track)

        qrels_path = Path(data_path).joinpath("qrels", f"{dataset_identifier.replace('/', '-')}-filtered-qrels.tsv")

        ClueWebQrel = namedtuple("ClueWebQrel", ['query_id', 'intent_id', 'doc_id', 'relevance'])

        # Create a list of named tuples that emulate ir_datasets structure
        original_qrels = []
        with open(qrels_path, "r") as qrels_file:
            for q_id, i_id, d_id, rel in csv.reader(qrels_file, delimiter="\t"):
                original_qrels.append(ClueWebQrel(q_id, i_id, d_id, rel))

        document_collections.append(original_qrels)

    else:
        dataset = ir_datasets.load(dataset_identifier)
        document_collections.append(dataset)

    annotation_duplicates = 0
    different_judgments = 0
    for dataset in document_collections:
        if "clueweb" in dataset_identifier or ("dl-mia" in dataset_identifier and intent_aware):
            totals = len(dataset)
            original_qrels = dataset
        else:
            original_qrels = dataset.qrels_iter()
            totals = dataset.qrels_count()

        # For each item in the QRels, retrieve the necessary database objects to enter a valid Annotation entry
        for original_label in tqdm(original_qrels, total=totals, desc=">> Loading human annotations..."):
            # For each item we need to see if there are matching database entries, which there may not be for some
            query = (Query.select()
                     .where((Query.q_id == original_label.query_id) &
                            (Query.dataset_name_id == dataset_db))
                     .get_or_none())
            document = (Document.select()
                        .where(Document.d_id == original_label.doc_id)
                        .get_or_none())

            if query is not None and document is not None:
                if intent_aware:
                    intent = (Intent.select()
                              .where((Intent.i_id == original_label.intent_id) &
                                     (Intent.query == query))
                              .get_or_none())
                    assert intent is not None
                    triple = (Triple.select()
                              .join_from(Triple, Query)
                              .join_from(Triple, Document)
                              .join_from(Triple, Intent)
                              .where((Triple.query == query) &
                                     (Triple.document == document) &
                                     (Triple.intent_id == intent.i_id))
                              .get_or_none())
                    try:
                        assert triple is not None
                    except AssertionError:
                        print("No triple was found")
                else:
                    triple = (Triple.select()
                              .join_from(Triple, Query)
                              .join_from(Triple, Document)
                              .where((Triple.query == query) &
                                     (Triple.document == document) &
                                     (Triple.intent.is_null()))
                              .get_or_none())

                if triple is not None:
                    # Create the Annotation entry
                    try:
                        Annotation.insert(triple=triple, config=config, result=original_label.relevance, error=None,
                                          truncated=False, explanation=None).execute()
                    except IntegrityError:
                        annotation_duplicates += 1
                        # retrieve the existing
                        existing = (Annotation.select().where((Annotation.triple_id == triple.id) & (Annotation.config_id == config.id)).get_or_none())
                        if existing is not None:
                            if existing.result != original_label.relevance:
                                different_judgments += 1
                            else:
                                Annotation.insert(triple=triple, config=config, result=original_label.relevance, error=None,
                                                truncated=False, explanation=None).on_conflict_ignore.execute()

    LOGGER.info(f"\tFinished loading human annotations for {dataset_identifier}.")
    LOGGER.info(f"\t{annotation_duplicates} duplicate annotations were found. {different_judgments} had different relevance scores.")
