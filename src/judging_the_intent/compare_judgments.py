import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import pandas as pd
from judging_the_intent.util.eval import Evaluator
from sklearn.metrics import classification_report, cohen_kappa_score
from scipy.stats import kstest, kendalltau

LOGGER = logging.getLogger(__file__)


def accuracy_for_agreement(u, v):
    assert len(u) == len(v)
    num_labels = len(u)
    identicals = 0
    for i in range(num_labels):
        if u[i] == v[i]:
            identicals += 1

    return identicals / num_labels


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


def build_combined_dataframe(human_df, with_intent, without_intent):
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

    combined_with_intent["rel"] = combined_with_intent.apply(get_human_annotations, args=(human_df,), axis=1)
    combined_without_intent["rel"] = combined_without_intent.apply(
        get_human_annotations, args=(human_df,), axis=1)

    LOGGER.info(f"combined_with_intent has {combined_with_intent['result'].isna().sum()} LLM items with NULL judgments")
    LOGGER.info(
        f"combined_with_intent has {combined_with_intent['rel'].isna().sum()} human items with NULL judgments")
    LOGGER.info(
        f"combined_without_intent has {combined_without_intent['result'].isna().sum()} LLM items with NULL judgments")
    LOGGER.info(
        f"combined_without_intent has {combined_without_intent['rel'].isna().sum()} LLM items with NULL judgments")

    return combined_with_intent, combined_without_intent


class JudgmentEvaluator(Evaluator):
    def __init__(self, model: str, data_dir: str, dataset: str) -> None:
        super().__init__(model, data_dir, dataset)

    def run(self) -> None:
        """ Run the evaluation

        Retrieves the Annotations for given Model and Dataset pair, then performs the evaluation
        """
        # Step 1 - Load original annotations (*-filtered-qrels.tsv)
        human_df = pd.read_csv(Path(self._data_dir).joinpath(
            f"{self._dataset.replace('/', '-')}-filtered-qrels.tsv"),
            sep="\t", names=["query_id", "intent_id", "doc_id", "rel"])
        # Filter out any rows that have negative relevance scores
        human_df = human_df[human_df["rel"] >= 0].copy()
        LOGGER.info(f"Loaded {human_df.shape[0]} human judgments.")

        # Step 2 - Retrieve all Annotation entries for a given Model, and filter those Annotations by Dataset

        # Get all Query objects for the current dataset
        with_intent, without_intent = self._retrieve_database_annotations()

        # Step 3 - Compare LLM to Original Human Annotation

        # Create a copy of human judgment DataFrame and add a column with the matching LLM Judgments
        combined_with_intent, combined_without_intent = build_combined_dataframe(human_df, with_intent, without_intent)

        # Classification accuracy
        with_intent_report = classification_report(combined_with_intent["rel"].values,
                                                   combined_with_intent["result"].values, labels=[0, 1, 2, 3])
        without_intent_report = classification_report(combined_without_intent["rel"].values,
                                                      combined_without_intent["result"].values,
                                                      labels=[0, 1, 2, 3])

        # Accuracy for Agreement Assessment
        agree_i = accuracy_for_agreement(combined_with_intent["rel"].values, combined_with_intent["result"].values)
        agree_no_i = accuracy_for_agreement(combined_without_intent["rel"].values, combined_without_intent["result"].values)

        # Kendall's Tau
        ktau_i = kendalltau(combined_with_intent["rel"].values, combined_with_intent["result"].values)
        ktau_no_i = kendalltau(combined_without_intent["rel"].values, combined_without_intent["result"].values)

        # KS Test
        ks_i = kstest(combined_with_intent["rel"].values, combined_with_intent["result"].values, alternative="greater")
        ks_no_i = kstest(combined_without_intent["rel"].values, combined_without_intent["result"].values,
                         alternative="greater")

        # Cohen's Kappa
        cohen_i = cohen_kappa_score(combined_with_intent["rel"].values, combined_with_intent["result"].values)
        cohen_no_i = cohen_kappa_score(combined_without_intent["rel"].values, combined_without_intent["result"].values)

        # Create the results directory if it doesn't exist already
        results_directory = Path(self._data_dir).parent.joinpath("compare-output")
        results_directory.mkdir(exist_ok=True)

        result_file_path = Path(results_directory).joinpath(
            f"{self._model.replace(':', '-')}-{self._dataset.replace('/', '-')}-classification-report.txt")
        LOGGER.info(f"Writing results to {result_file_path}")

        with (open(result_file_path, "w") as result_file):
            result_file.write("CLASSIFICATION ACCURACY WITH INTENT\n\n")
            result_file.write(with_intent_report)
            result_file.write("\n\nCLASSIFICATION ACCURACY WITHOUT INTENT\n\n")
            result_file.write(without_intent_report)
            result_file.write("\n\nKENDALL'S TAU WITH INTENT\n\n")
            result_file.write(f"Statistic:\t{ktau_i.statistic}\nP-Value:\t{ktau_i.pvalue}")
            result_file.write("\n\nKENDALL's TAU WITHOUT INTENT\n\n")
            result_file.write(f"Statistic:\t{ktau_no_i.statistic}\nP-Value:\t{ktau_no_i.pvalue}")
            result_file.write("\n\nKOLMOGOROV-SMIRNOV TEST WITH INTENT\n\n")
            result_file.write(f"Statistic:\t{ks_i.statistic}\nP-Value:\t{ks_i.pvalue}")
            result_file.write("\n\nKOLMOGOROV-SMIRNOV TEST WITHOUT INTENT\n\n")
            result_file.write(f"Statistic:\t{ks_no_i.statistic}\nP-Value:\t{ks_no_i.pvalue}")
            result_file.write("\n\nCOHEN'S KAPPA WITH INTENT\n\n")
            result_file.write(f"Kappa:\t{cohen_i}")
            result_file.write("\n\nCOHEN'S KAPPA WITHOUT INTENT\n\n")
            result_file.write(f"Kappa:\t{cohen_no_i}")
            result_file.write("\n\nACCURACY FOR AGREEMENT WITH INTENT\n\n")
            result_file.write(f"Accuracy:\t{agree_i}")
            result_file.write("\n\nACCURACY FOR AGREEMENT WITHOUT INTENT\n\n")
            result_file.write(f"Accuracy:\t{agree_no_i}")


def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--models", required=True, nargs="+", help="Ollama model identifiers."
    )
    ap.add_argument("-d", dest="datasets", required=True, nargs="+", help="Dataset identifiers")
    ap.add_argument("--data_dir", help="Directory containing the TREC Qrels files.",
                    default=str(Path(__file__).parent.parent.parent.joinpath("trec-web", "qrels")))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    for model in args.models:
        for dataset in args.datasets:
            LOGGER.info(f"Evaluating {model} annotations of {dataset}.")
            JudgmentEvaluator(model, args.data_dir, dataset).run()
        break

if __name__ == "__main__":
    main()