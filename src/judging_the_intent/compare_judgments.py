import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report

from judging_the_intent import __version__
from judging_the_intent.db.schema import (
    Annotation,
    Config,
    Query,
    Triple,
)

LOGGER = logging.getLogger(__file__)

def get_human_annotations(row, human_df):
    # Check both LLM dataframes for the matching triple, and retrieve its annotation
    if isinstance(row["intent_id"], str):
        subframe = human_df[
            (human_df["query_id"] == row["query_id"]) &
            (human_df["doc_id"] == row["doc_id"])
        ]
    else:
        subframe = human_df[
            (human_df["query_id"] == row["query_id"]) &
            (human_df["intent_id"] == row["intent_id"]) &
            (human_df["doc_id"] == row["doc_id"])
        ]

    if not subframe.empty:
        first = subframe.head(1)
        return first["rel"].values[0]


class Evaluator:
    def __init__(self, model: str, data_dir: str, dataset: str, runtype=None) -> None:
        self._model = model
        self._dataset = dataset
        self._runtype = runtype # Options are [None (default) | ranker]
        self._data_dir = data_dir

    def run(self) -> None:
        """ Run the evaluation

        Retrieves the Annotations for given Model and Dataset pair, then performs the evaluation
        """
        config, created = Config.get_or_create(
            model_name=self._model, version=__version__
        )
        if created:
            LOGGER.info(
                "model %s (version %s) not found in DB, creating",
                self._model,
                __version__,
            )
        else:
            LOGGER.info("found model %s (version %s) in DB", self._model, __version__)
        # Step 1 - Load original annotations (*-filtered-qrels.tsv)
        human_df = pd.read_csv(Path(__file__).parent.parent.parent.parent.joinpath("trec-web", "qrels",
                                                                    f"{self._dataset.replace('/', '-')}-filtered-qrels.tsv"),
                               sep="\t", names=["query_id", "intent_id", "doc_id", "rel"])
        # Filter out any rows that have negative relevance scores
        human_df = human_df[human_df["rel"] >= 0].copy()
        LOGGER.info(f"Loaded {human_df.shape[0]} human judgments.")

        # Step 2 - Retrieve all Annotation entries for a given Model, and filter those Annotations by Dataset

        # Get all Query objects for the current dataset
        dataset_queries = (
            Query.select()
            .where(Query.dataset_name == self._dataset)
            .alias("dataset_queries")
        )

        # Get all Triple objects that have ForeignKey relationships to the dataset Query objects, that have Intents
        dataset_triples_with_intent = (
            Triple.select()
            .where(Triple.intent.is_null(False))
            .join(dataset_queries, on=(Triple.query == dataset_queries.c.q_id))
            .join_from(Triple, Query)
        )

        # Get all Triple objects that have ForeignKey relationships to the dataset Query objects, that do not have Intents
        dataset_triples_without_intent = (
            Triple.select()
            .where(Triple.intent.is_null())
            .join(dataset_queries, on=(Triple.query == dataset_queries.c.q_id))
            .join_from(Triple, Query)
        )

        # Collect the related Annotation objects for the Triple entries, with and without intent
        model_annotations_with_intent = (
            Annotation.select(
                Annotation,
                Triple.query.alias("query_id"),
                Triple.intent.alias("intent_id"),
                Triple.document.alias("doc_id"),
            )
            .join(Config, on=(Annotation.config == config.id))
            .join_from(Annotation, dataset_triples_with_intent, on=(Annotation.triple == dataset_triples_with_intent.c.id))
            .join_from(Annotation, Triple)
        )

        model_annotations_without_intent = (
            Annotation.select(
                Annotation,
                Triple.query.alias("query_id"),
                Triple.intent.alias("intent_id"),
                Triple.document.alias("doc_id"),
            )
            .join(Config, on=(Annotation.config == config.id))
            .join_from(Annotation, dataset_triples_without_intent,
                       on=(Annotation.triple == dataset_triples_without_intent.c.id))
            .join_from(Annotation, Triple)
        )

        with_intent = pd.DataFrame(model_annotations_with_intent.dicts())
        without_intent = pd.DataFrame(model_annotations_without_intent.dicts())
        LOGGER.info(f"Loaded {with_intent.shape[0]} LLM judgments with intent and {without_intent.shape[0]} without intent.")

        # Step 3 - Compare LLM to Original Human Annotation

        # Create a copy of human judgment DataFrame and add a column with the matching LLM Judgments
        combined_with_intent = with_intent.copy()
        combined_with_intent = combined_with_intent.dropna(subset=["result"])
        combined_without_intent = without_intent.copy()
        combined_without_intent = combined_without_intent.dropna(subset=["result"])

        # Ensure all judgments and IDs (except doc_id) are int64
        combined_with_intent[["query_id", "intent_id", "result"]] = combined_with_intent[
            ["query_id", "intent_id", "result"]].astype("Int64")
        combined_without_intent[["query_id", "result"]] = combined_without_intent[
            ["query_id", "result"]].astype("Int64")
        combined_without_intent["intent_id"] = combined_without_intent["intent_id"].fillna('')
        human_df[["query_id", "intent_id", "rel"]] = human_df[["query_id", "intent_id", "rel"]].astype("Int64")

        combined_with_intent["rel"] = combined_with_intent.apply(get_human_annotations, args=(human_df,),
                                                                            axis=1)
        combined_without_intent["rel"] = combined_without_intent.apply(
            get_human_annotations, args=(human_df,), axis=1)

        LOGGER.info(f"combined_with_intent has {combined_with_intent['result'].isna().sum()} LLM items with NULL judgments")
        LOGGER.info(
            f"combined_with_intent has {combined_with_intent['rel'].isna().sum()} human items with NULL judgments")
        LOGGER.info(f"combined_without_intent has {combined_without_intent['result'].isna().sum()} LLM items with NULL judgments")
        LOGGER.info(
            f"combined_without_intent has {combined_without_intent['rel'].isna().sum()} LLM items with NULL judgments")

        if self._runtype == "ranker":
            print(f"Ranker runtype executed for\nModel:\t\t{self._model}\nDataset:\t{self._dataset}")
            from judging_the_intent.util.rankers import run_rankers
            print(run_rankers(self._dataset, self._data_dir, combined_with_intent, combined_without_intent))
        else:
            with_intent_report = classification_report(combined_with_intent["rel"].values,
                                                       combined_with_intent["result"].values, labels=[0, 1, 2, 3])
            without_intent_report = classification_report(combined_without_intent["rel"].values,
                                                          combined_without_intent["result"].values,
                                                          labels=[0, 1, 2, 3])

            # Create the results directory if it doesn't exist already
            results_directory = Path.cwd().joinpath("results")
            results_directory.mkdir(exist_ok=True)

            result_file_path = Path(__file__).parent.parent.joinpath(
                f"{self._model}-{self._dataset.replace('/', '-')}-classification-report.txt")
            LOGGER.info(f"Writing results to {result_file_path}")

            with (open(result_file_path, "w") as result_file):
                result_file.write("WITH INTENT\n\n")
                result_file.write(with_intent_report)
                result_file.write("\n\nWITHOUT INTENT\n\n")
                result_file.write(without_intent_report)


def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--models", required=True, nargs="+", help="Ollama model identifiers."
    )
    ap.add_argument("--datasets", required=True, nargs="+", help="Dataset identifiers")
    ap.add_argument("-r", dest="runtype", action="store_true", default=False,
                    help="Flag to trigger ranker execution. Only uses the first model and dataset provided. All others are ignored.")
    ap.add_argument("--data_dir", help="Directory containing the TREC Qrels files.",
                    default=str(Path(__file__).parent.parent.parent.parent.joinpath("results", "clueweb")))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.runtype:
        for dataset in args.datasets:
            Evaluator(args.models[0], args.data_dir, dataset, "ranker").run()
    else:
        for model in args.models:
            for dataset in args.datasets:
                LOGGER.info(f"Evaluating {model} annotations of {dataset}.")
                Evaluator(model, dataset).run()

if __name__ == "__main__":
    main()