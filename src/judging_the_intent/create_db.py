import csv
import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from tqdm import tqdm

import ir_datasets
from ir_datasets_subsample import register_subsamples

from peewee import fn
from judging_the_intent.db import DATABASE
from judging_the_intent.db.schema import (
    Annotation,
    Config,
    Document,
    Intent,
    Query,
    Triple,
)

LOGGER = logging.getLogger(__file__)


def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--datasets", nargs="+", required=True, help="List of dataset identifiers."
    )
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path(__file__).parent.parent.parent.joinpath("datasets"),
        help="Where dataset files are located.",
    )
    ap.add_argument("--intent", action="store_true", default=False, help="Include search intents in Triples creation")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)
    register_subsamples()

    with DATABASE:
        LOGGER.info("Creating initial tables...")
        DATABASE.create_tables([Query, Intent, Document, Triple, Config, Annotation])

    for dataset_name in args.datasets:
        LOGGER.info("\tprocessing %s", dataset_name)
        try:
            if "dl-mia" in dataset_name:
                # Need to load msmarco-passage-v2
                dataset = ir_datasets.load("msmarco-passage-v2")
                docs_store = dataset.docs_store()
            else:
                dataset = ir_datasets.load(dataset_name)
                docs_store = dataset.docs_store()
        except Exception as e:
            LOGGER.exception("failed to load dataset %s", dataset_name)
            raise e
        qd_pairs = set()

        # Create the special branch for ClueWeb subsample
        if "clueweb" in dataset_name or "dl-mia" in dataset_name:
            if "clueweb" in dataset_name:
                dataset_name_split = dataset_name.split("/")
                dataset_top_level_name = dataset_name_split[1]
                dataset_track = dataset_name_split[-1]
                data_path = Path(args.data_dir).joinpath(dataset_top_level_name, dataset_track)
                query_path = Path(data_path).joinpath("queries", f"{dataset_name.replace('/', '-')}-queries.tsv")
                qrels_path = Path(data_path).joinpath("qrels", f"{dataset_name.replace('/', '-')}-filtered-qrels.tsv")
                intent_path = Path(data_path).joinpath("intent", f"{dataset_name.replace('/', '-')}-qid-iid-intent.tsv")
            else: # DL-MIA branch
                data_path = Path(args.data_dir).joinpath(dataset_name)
                query_path = Path(data_path).joinpath("queries", f"{dataset_name}.queries.tsv")
                intent_path = Path(data_path).joinpath("intent", f"{dataset_name}.intents.tsv")
                qrels_path = Path(data_path).joinpath("qrels", f"{dataset_name}.qid_iid_qrel.txt")

            # Load the Queries
            with open(str(query_path), encoding="utf-8", newline="", ) as fp:
                for q_id, q_text in tqdm(csv.reader(fp, delimiter="\t"), desc=">> Inserting Queries..."):
                    Query.insert(q_id=q_id, dataset_name=dataset_name, text=q_text).on_conflict(
                        conflict_target=[Query.q_id], preserve=[Query.q_id],
                        update={Query.dataset_name: fn.CONCAT(Query.dataset_name, f", {dataset_name}")},
                        where=(~Query.dataset_name.contains(dataset_name))
                    ).execute()

            # Load the Intents
            with open(str(intent_path), encoding="utf-8", newline="", ) as fp:
                for q_id, i_id, i_text in tqdm(csv.reader(fp, delimiter="\t"), desc=">> Inserting Intents..."):
                    Intent.create(i_id=i_id, query=q_id, text=i_text)

            delim = " " if "dl-mia" in dataset_name else "\t"
            with open(str(qrels_path), encoding="utf-8", newline="") as fp:
                for q_id, i_id, d_id, rel in csv.reader(fp, delimiter=delim):
                    # QRels should be filtered already
                    try:
                        assert int(rel) >= 0
                    except AssertionError:
                        LOGGER.warning("%s has invalid relevance score of %s", d_id, rel)
                        continue

                    # keep track of unique query-doc pairs to add a null-intent triple for each one later
                    qd_pairs.add((q_id, d_id))

                    try:
                        d_text = docs_store.get(d_id).text
                    except KeyError:
                        LOGGER.warning("%s not found in document store", d_id)
                        d_text = ""

                    # we expect duplicates here
                    Document.insert(d_id=d_id, text=d_text).on_conflict_ignore().execute()

                    Triple.create(query=q_id, intent=i_id, document=d_id)

            for q_id, d_id in qd_pairs:
                Triple.create(query=q_id, intent=None, document=d_id)

        else:
            queries = set()
            qid_pairs = set()
            for query in tqdm(dataset.queries_iter(), total=dataset.queries_count(), desc=">> Inserting Queries..."):
                queries.add(query.query_id)
                Query.insert(q_id=query.query_id, dataset_name=dataset_name, text=query.text).on_conflict_ignore().execute()
            dataset_top_level_name, dataset_track = dataset_name.split("/")
            filtered_qrels_directory_path = Path(args.data_dir).joinpath(dataset_top_level_name, dataset_track, "qrels")
            filtered_qrels_directory_path.mkdir(parents=True, exist_ok=True)
            with open(str(filtered_qrels_directory_path.joinpath(f"{dataset_track}-filtered-qrels.tsv")),
                encoding="utf-8",
                newline="",
                mode="w"
            ) as fp:
                for qrel in tqdm(dataset.qrels_iter(), total=dataset.qrels_count(), desc=">> Inserting QRels..."):
                    # namedtuple<query_id, doc_id, relevance, iteration>
                    if qrel.query_id in queries:
                        qid_pairs.add((qrel.query_id, qrel.doc_id))
                        try:
                            d_text = docs_store.get(qrel.doc_id).text
                        except KeyError:
                            LOGGER.warning("%s not found in document store", qrel.doc_id)
                            d_text = ""

                        # we expect duplicates here
                        Document.insert(d_id=qrel.doc_id, text=d_text).on_conflict_ignore().execute()
                        Triple.create(query=qrel.query_id, intent=None, document=qrel.doc_id)

                        # Also build the filtered-qrels .csv for evaluation purposes. Future versions will convert this to DB driven
                        fp.write(f"{qrel.query_id}\t0\t{qrel.doc_id}\t{qrel.relevance}\n")

if __name__ == "__main__":
    main()