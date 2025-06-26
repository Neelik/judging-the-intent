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
    # Check both LLM dataframes for the matching triple, and retrieve it's annotation
    subframe = human_df[
        (human_df["query_id"] == row["query_id"]) &
        (human_df["intent_id"] == row["intent_id"]) &
        (human_df["doc_id"] == row["doc_id"])
    ]

    if not subframe.empty:
        return subframe["rel"]


class Evaluator:
    def __init__(self, model: str, dataset: str) -> None:
        self._model = model
        self._dataset = dataset
        self._best = None

    def run(self):
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
        queries_subquery = Query.select().where(Query.dataset_name == self._dataset)
        triples_subquery = Triple.select().where(Triple.query.in_(queries_subquery))

        all_annotations_cte = (
            Annotation.select(
                Annotation.triple,
                Annotation.config,
                Annotation.result,
                Triple.query.alias("query_id"),
                Triple.intent.alias("intent_id"),
                Triple.document.alias("doc_id")
            )
            .join(Config)
            .join(Triple)
            .where(Config.id == config.id)
            .where(Annotation.triple.in_(triples_subquery))
            .cte("all_annotations")
        )

        model_annotations_without_intent = (
            all_annotations_cte.select_from(
                all_annotations_cte.triple,
                all_annotations_cte.config,
                all_annotations_cte.result,
                all_annotations_cte.query_id,
                all_annotations_cte.intent_id,
                all_annotations_cte.doc_id,
            )
            .where(all_annotations_cte.triple.in_(Triple.intent.is_null()))
        )

        model_annotations_with_intent = (
            all_annotations_cte.select_from(
                all_annotations_cte.triple,
                all_annotations_cte.config,
                all_annotations_cte.result,
                all_annotations_cte.query_id,
                all_annotations_cte.intent_id,
                all_annotations_cte.doc_id,
            )
            .where(all_annotations_cte.triple.in_(Triple.intent.is_null(False)))
        )
        with_intent = pd.DataFrame(model_annotations_with_intent.dicts())
        without_intent = pd.DataFrame(model_annotations_without_intent.dicts())
        LOGGER.info(f"Loaded {human_df.shape[0]} LLM judgments.")

        # Step 3 - Compare LLM to Original Human Annotation

        # Create a copy of human judgment DataFrame and add a column with the matching LLM Judgments
        combined_with_intent = with_intent.copy()
        combined_without_intent = without_intent.copy()
        combined_with_intent["rel"] = combined_with_intent.apply(get_human_annotations, args=(human_df,),
                                                                            axis=1)

        combined_without_intent["rel"] = combined_without_intent.apply(
            get_human_annotations, args=(human_df,), axis=1)

        with_intent_report = classification_report(combined_with_intent["rel"].values,
                                                   combined_with_intent["llm_annotation"].values, labels=[0, 1, 2, 3])
        without_intent_report = classification_report(combined_without_intent["rel"].values,
                                                      combined_without_intent["llm_annotation"].values,
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
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    for model in args.models:
        for dataset in args.datasets:
            LOGGER.info(f"Evaluating {model} annotations of {dataset}.")
            Evaluator(model, dataset).run()

if __name__ == "__main__":
    main()