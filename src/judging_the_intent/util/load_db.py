import csv
import ir_datasets
import pandas as pd
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from datetime import datetime
from ir_datasets.indices import Docstore
from ir_datasets.formats import TrecQrel, GenericQuery
from ir_datasets_subsample import register_subsamples
from collections.abc import Iterable
from typing import Optional
from sentence_transformers import SentenceTransformer, util
from peewee import ProgrammingError
from tqdm import tqdm
from pathlib import Path

from judging_the_intent.db.schema import (
    Dataset,
    Document,
    Intent,
    Query,
    Triple,
)

import logging
LOGGER = logging.getLogger(__name__)


def extract_timestamp(file_path) -> int:
    """
        Extract timestamp from file name

        :param file_path: Path to the file to be parsed
        :return: Timestamp as int
    """
    call_time = datetime.now().timestamp()
    try:
        filename = file_path.name
        timestamp_str = filename.split('_')[0]
        return int(timestamp_str)
    except (ValueError, IndexError):
        return int(call_time)


def get_latest_file_by_timestamp(directory: Path, pattern: str="*generated_*.tsv") -> Path:
    files = directory.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern} in directory {directory}.")

    latest_file = max(files, key=extract_timestamp)
    return latest_file


def load_text_data(dataset_identifier: str, docs_store: Docstore, query_path: Path, intent_path: Path, qrels_path: Path) -> bool:
    """
        Function to read text data, usually in the form of TREC Qrels, and load it into the database

        :param dataset_identifier: Name of the Dataset
        :param docs_store: Docstore object from the loaded ir_dataset
        :param query_path: Path to the query file
        :param intent_path: Path to the intent file
        :param qrels_path: Path to the qrels file

        :return: True if data was loaded, False otherwise
    """
    qd_pairs = set()
    # Load the Queries
    with (open(str(query_path), encoding="utf-8", newline="", ) as fp):
        for q_id, q_text in tqdm(csv.reader(fp, delimiter="\t"), desc=">> Inserting Queries..."):
            Query.insert(q_id=q_id, dataset_name=dataset_identifier, text=q_text).execute()

    # Load the Intents
    with open(str(intent_path), encoding="utf-8", newline="", ) as fp:
        for q_id, i_id, i_text in tqdm(csv.reader(fp, delimiter="\t"), desc=">> Inserting Intents..."):
            # Retrieve the current Query
            query = (Query.select()
                     .where(Query.q_id == q_id)
                     .where(Query.dataset_name == dataset_identifier)
                     )
            Intent.insert(i_id=i_id, query=query, text=i_text).on_conflict_ignore().execute()

    delim = " " if "dl-mia" in dataset_identifier else "\t"
    invalid_rels = 0
    missing_docs = 0
    with open(str(qrels_path), encoding="utf-8", newline="") as fp:
        for q_id, i_id, d_id, rel in tqdm(csv.reader(fp, delimiter=delim), desc=">> Inserting Documents and Triples..."):
            # QRels should be filtered already
            try:
                assert int(rel) >= 0
            except AssertionError:
                invalid_rels += 1
                continue

            # keep track of unique query-doc pairs to add a null-intent triple for each one later
            qd_pairs.add((q_id, d_id))

            try:
                d_text = docs_store.get(d_id).text
            except KeyError:
                missing_docs += 1
                d_text = ""

            # we expect duplicates here
            Document.insert(d_id=d_id, text=d_text.replace("\t", " ").replace("\n", " ").replace("\r",
                                                                                                 " ")).on_conflict_ignore().execute()

            query = (Query.select().where(Query.q_id == q_id).where(Query.dataset_name == dataset_identifier))
            Triple.create(query=query, intent=i_id, document=d_id)

    for q_id, d_id in qd_pairs:
        query = (Query.select().where(Query.q_id == q_id).where(Query.dataset_name == dataset_identifier))
        Triple.create(query=query, intent=None, document=d_id)

    LOGGER.info(f"\tDocuments with invalid relevance scores: {invalid_rels}")
    LOGGER.info(f"\tDocuments with missing content: {missing_docs}")
    return True


def load_clueweb(dataset_identifier: str, data_dir: str) -> None:
    """
        Special function to load TREC Web datasets using ClueWeb subsamples into the database

        :param dataset_identifier: Name of the Dataset
        :param data_dir: Path to the datasets directory
    """
    register_subsamples()
    dataset = ir_datasets.load(dataset_identifier)
    dataset_name_split = dataset_identifier.split("/")
    dataset_top_level_name = dataset_name_split[1]
    dataset_track = dataset_name_split[-1]
    data_path = Path(data_dir).joinpath(dataset_top_level_name, dataset_track)
    query_path = Path(data_path).joinpath("queries", f"{dataset_identifier.replace('/', '-')}-queries.tsv")
    qrels_path = Path(data_path).joinpath("qrels", f"{dataset_identifier.replace('/', '-')}-filtered-qrels.tsv")
    intent_path = Path(data_path).joinpath("intent", f"{dataset_identifier.replace('/', '-')}-qid-iid-intent.tsv")

    loaded = load_text_data(dataset_identifier, dataset.docs_store(), query_path, intent_path, qrels_path)
    LOGGER.info(f"\tDataset {dataset_identifier} loaded: {loaded}")


def load_dl_mia(dataset_identifier: str, data_dir: str) -> None:
    """
        Special function to load DL-MIA dataset into the database

        :param dataset_identifier: Name of the Dataset
        :param data_dir: Path to the datasets directory
    """
    # Need to load the full msmarco-passage-v2 for access to both trec-dl-2021 trec-dl-2022
    dataset = ir_datasets.load("msmarco-passage-v2")
    data_path = Path(data_dir).joinpath(dataset_identifier)
    query_path = Path(data_path).joinpath("queries", f"{dataset_identifier}.queries.tsv")
    intent_path = Path(data_path).joinpath("intent", f"{dataset_identifier}.intents.tsv")
    qrels_path = Path(data_path).joinpath("qrels", f"{dataset_identifier}.qid_iid_qrel.txt")

    loaded = load_text_data(dataset_identifier, dataset.docs_store(), query_path, intent_path, qrels_path)
    LOGGER.info(f"\tDataset {dataset_identifier} loaded: {loaded}")


def load_ir_dataset(dataset_identifier: str, intent: bool) -> None:
    """
        Generic method to load known ir_datasets into the database. Not tested on all supported datasets.

        :param dataset_identifier: Name of the Dataset
        :param intent: Whether the intent should be loaded
    """
    dataset = ir_datasets.load(dataset_identifier)
    queries_loaded = load_queries(dataset_identifier, dataset.queries_iter(), dataset.queries_count())
    if intent:
        # Load the intents into the database
        load_ird_intents(dataset_identifier, Path.cwd())

    load_qrels(dataset_identifier, dataset.docs_store(), dataset.qrels_iter(), dataset.qrels_count(), queries_loaded)


def load_qrels(dataset_identifier, docs_store: Docstore,
               q_iter: Iterable[TrecQrel], q_count: int, query_ids: set) -> None:
    """
        Function to load qrel data into the database.

        :param dataset_identifier: Name of the Dataset
        :param docs_store: Docstore object from ir_datasets for the specified dataset
        :param q_iter: Iterable of TrecQrel instances from ir_datasets
        :param q_count: Number of queries to load
        :param query_ids: Set of query IDs already loaded into database
        :return: True if qrels were loaded, False otherwise
    """
    for qrel in tqdm(q_iter, total=q_count, desc=">> Inserting QRels..."):
        if qrel.query_id in query_ids:
            try:
                d_text = docs_store.get(qrel.doc_id).text
            except KeyError:
                LOGGER.debug("\t%s not found in document store", qrel.doc_id)
                d_text = ""

            # we expect duplicates here
            Document.insert(d_id=qrel.doc_id, text=d_text.replace(
                "\t", " ").replace("\n", " ").replace("\r"," ")).on_conflict_ignore().execute()
            query = (Query.select().where(Query.q_id == qrel.query_id).where(Query.dataset_name_id == dataset_identifier))
            try:
                Triple.create(query=query, intent=None, document=qrel.doc_id)
            except ProgrammingError as e:
                LOGGER.error(f"{e.__traceback__}\n\n{query}")


def load_queries(dataset_identifier, q_iter: Iterable[GenericQuery], q_count: int) -> set:
    """
        Function to load query data into the database.

        :param dataset_identifier: Name of the Dataset
        :param q_iter: Iterable of GenericQuery instances from ir_datasets
        :param q_count: Number of queries to load
        :return: A set of the query IDs loaded
    """
    queries = set()
    for query in tqdm(q_iter, total=q_count, desc=">> Inserting Queries..."):
        queries.add(query.query_id)
        Query.insert(q_id=query.query_id,
                     dataset_name=dataset_identifier,
                     text=query.text).on_conflict_ignore().execute()

    return queries


def load_ird_intents(dataset_identifier: str, intent_path: Optional[Path] = None) -> bool:
    """
        Function to load LLM-generated intent data into the database.

        :param dataset_identifier: Name of the Dataset
        :param intent_path: Path object giving the filepath to the intents
        :return: True if intents were loaded, False otherwise
    """
    raise NotImplemented("Intents are not currently supported in ir_datasets.")

def load_generated_intents(dataset_identifier: str, generation_type: str, intent_path: Path) -> bool:
    """
        Function to load LLM-generated intent data into the database.

        :param dataset_identifier: Name of the Dataset
        :param generation_type: Type of intent generation being used, options being intent or subtopic
        :param intent_path: Path object giving the filepath to the intents
        :return: True if intents were loaded, False otherwise
    """
    assert generation_type in ("intent", "subtopic")

    # Load the generated intents
    generated_intents = pd.read_csv(Path(intent_path), sep="\t")
    generated_intents = generated_intents[generated_intents.document_text != ""]

    if "similarity" not in generated_intents.columns:
        # Determine sentence similarity between the query and the generated intent/subtopic

        sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        tqdm.pandas(desc="Calculating query-intent similarity...")
        generated_intents["similarity"] = generated_intents.progress_apply(
            lambda row: util.pytorch_cos_sim(sim_model.encode(row["query_text"], convert_to_tensor=True,
                                                              show_progress_bar=False),
                                             sim_model.encode(row["intent_text"], convert_to_tensor=True,
                                                              show_progress_bar=False)), axis=1)

        tqdm.pandas(desc="Extracting similarity scores...")
        generated_intents["similarity"] = generated_intents["similarity"].progress_apply(
            lambda tens: tens.cpu().detach().numpy().flatten()[0])
        generated_intents.to_csv(Path(intent_path), sep="\t", index=False)

    # Grab ones that are of medium to medium-high similarity. We want intents/subtopics that are semantically similar
    # enough to be related, but dissimilar enough to be distinct and diverse.
    fuzzy_matches = generated_intents[(generated_intents["similarity"] <= 0.85) & (generated_intents["similarity"] >= 0.75)]

    # Iterate over the unique query_ids, and sample 5 intents/subtopics for each query
    unique_qids = fuzzy_matches.query_id.unique()
    db_writable = []
    for qid in unique_qids:
        subframe = fuzzy_matches[fuzzy_matches["query_id"] == qid]
        sampled = subframe.sample(n=min(5, subframe.shape[0]), random_state=42)
        records = sampled.to_dict("records")
        db_writable.extend(records)

    total_insertions = 0
    if generation_type == "subtopic":
        # For every entry in db_writable, retrieve the Query and the related Triples (to get all documents for the query)
        for pos, record in tqdm(enumerate(db_writable), total=len(db_writable), desc=">> Writing new intents to database..."):
            query = (Query.select().where((Query.dataset_name_id == dataset_identifier) & (Query.id == record["query_id"])).get_or_none())
            if query is not None:
                triples = (
                    Triple.select()
                    .where((Triple.query == query) & (Triple.intent.is_null()))
                )
                triples_tups = triples.namedtuples()
                documents = [tt.document for tt in triples_tups]
                triples_for_db = []

                # Create the Intent entry
                intent_db = Intent.create(i_id=f"gen_{generation_type}_{pos}", query=query, text=record["intent_text"],
                                          source=f"generated-{generation_type}")

                [triples_for_db.append({"intent": intent_db, "query_id": record["query_id"], "document_id": d}) for d in documents]
                # Create the Triple entries
                inserted = Triple.insert_many(triples_for_db).on_conflict_ignore().as_rowcount().execute()
                total_insertions += inserted

            else:
                LOGGER.debug(f"Query {record['query_id']} not found in database.")
                continue

        LOGGER.info(f"Inserted {total_insertions} Triples.")
        return total_insertions > 0

    else:
        # Handling for generated-intent will go here in the future
        return False

def main():
    """
    Driver function that parses command line arguments and calls appropriate functions.

    """
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--datasets", nargs="+", required=True, help="List of dataset identifiers."
    )
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).parent.parent.parent.parent.joinpath("datasets"),
        help="Where dataset files are located.",
    )
    ap.add_argument("--intent", action="store_true", default=False, help="Include search intents in Triples creation")
    ap.add_argument("--load_generated_intents", action="store_true", default=False, help="Load generated intents/subtopics into the database")
    ap.add_argument("--generation_type", type=str, required=False, choices=("subtopic", "intent"), help="Defines the generation type: 'subtopic' or 'intent'")
    ap.add_argument("--intent_dir", type=Path, default=None, help="Path to the directory containing intent file.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    # This branch handles loading generated intents directly into the database
    if args.load_generated_intents:
        if args.intent:
            LOGGER.warning("\t--intent is ignored and has no effect when --load_generated_intents is True.")

        # Check that a path to the intents to be loaded exists, i.e., have they been generated already?
        if args.intent_dir is None:
            LOGGER.warning(f"\t--intent_dir not set. Path to directory will be inferred by dataset identifier provided in --datasets. To prevent this warning, pass a valid path to --intent_dir.")

        # Check if the generation type is set
        if args.generation_type is None:
            raise RuntimeError("--generation_type is required when --load_generated_intents is True.")

        for dataset_identifier in args.datasets:
            if args.intent_dir is None:
                # Infer the intent path if necessary
                dataset_name_split = dataset_identifier.split("/")
                dataset_top_level_name = dataset_name_split[1]
                dataset_track = dataset_name_split[-1]
                data_path = Path(args.data_dir).joinpath(dataset_top_level_name, dataset_track, "intent")
            else:
                data_path = args.intent_dir

            latest_intents_file = get_latest_file_by_timestamp(data_path)

            load_generated_intents(dataset_identifier=dataset_identifier, intent_path=latest_intents_file,
                                   generation_type=args.generation_type)

    else:
        # This branch handles loading a dataset from ir_datasets into the database
        for dataset_identifier in args.datasets:
            LOGGER.info(f"\tLoading dataset {dataset_identifier}...")
            try:
                Dataset.create(name=dataset_identifier)
            except ProgrammingError as p_error:
                LOGGER.error(f"\t{str(p_error)}Did you run create_db?")
                break
            if "clueweb" in dataset_identifier:
                load_clueweb(dataset_identifier, args.data_dir)
            elif "dl-mia" in dataset_identifier:
                load_dl_mia(dataset_identifier, args.data_dir)
            else:
                load_ir_dataset(dataset_identifier, args.intent)


if __name__ == "__main__":
    main()