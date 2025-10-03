import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
import pandas as pd
from judging_the_intent.util.eval import Evaluator
from sklearn.metrics import classification_report, cohen_kappa_score, accuracy_score

LOGGER = logging.getLogger(__file__)


def accuracy_for_agreement(u, v):
    assert len(u) == len(v)
    num_labels = len(u)
    if num_labels == 0:
        return 0.0
    else:
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

    # Get the unique query-doc pairs, based on human
    qd_pairs = set()
    for row in human_df.iterrows():
        row = row[1]
        qd_pairs.add((row["query_id"], row["doc_id"]))

    # Determine how many repeats of judgments are necessary to match without intent to with intent
    for pair in qd_pairs:
        subframe = human_df[
            (human_df["query_id"] == pair[0]) &
            (human_df["doc_id"] == pair[1])
        ]
        # Get the row in without_intent
        wi_row = combined_without_intent[
            (combined_without_intent["query_id"] == pair[0]) &
            (combined_without_intent["doc_id"] == pair[1])
        ]
        temp_df = pd.DataFrame(wi_row)
        if subframe.shape[0] > 1:
            temp_df_expanded = pd.concat([temp_df] * (subframe.shape[0] - 1), ignore_index=True)
        else:
            temp_df_expanded = temp_df
        combined_without_intent = pd.concat([combined_without_intent, temp_df_expanded], ignore_index=True)

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

        # Label 0
        combined_with_intent_zero = combined_with_intent[combined_with_intent["rel"] == 0]
        combined_without_intent_zero = combined_without_intent[combined_without_intent["rel"] == 0]
        class_zero_with_intent = accuracy_for_agreement(combined_with_intent_zero["rel"].values,
                                                        combined_with_intent_zero["result"].values)
        class_zero_without_intent = accuracy_for_agreement(combined_without_intent_zero["rel"].values,
                                                           combined_without_intent_zero["result"].values)
        sk_class_zero_with_intent = accuracy_score(combined_with_intent_zero["rel"].values,
                                                   combined_with_intent_zero["result"].values)
        sk_class_zero_without_intent = accuracy_score(combined_without_intent_zero["rel"].values,
                                                      combined_without_intent_zero["result"].values)

        # Label 1
        combined_with_intent_one = combined_with_intent[combined_with_intent["rel"] == 1]
        combined_without_intent_one = combined_without_intent[combined_without_intent["rel"] == 1]
        class_one_with_intent = accuracy_for_agreement(combined_with_intent_one["rel"].values,
                                                        combined_with_intent_one["result"].values)
        class_one_without_intent = accuracy_for_agreement(combined_without_intent_one["rel"].values,
                                                           combined_without_intent_one["result"].values)
        sk_class_one_with_intent = accuracy_score(combined_with_intent_one["rel"].values,
                                                  combined_with_intent_one["result"].values)
        sk_class_one_without_intent = accuracy_score(combined_without_intent_one["rel"].values,
                                                     combined_without_intent_one["result"].values)

        # Label 2
        combined_with_intent_two = combined_with_intent[combined_with_intent["rel"] == 2]
        combined_without_intent_two = combined_without_intent[combined_without_intent["rel"] == 2]
        class_two_with_intent = accuracy_for_agreement(combined_with_intent_two["rel"].values,
                                                        combined_with_intent_two["result"].values)
        class_two_without_intent = accuracy_for_agreement(combined_without_intent_two["rel"].values,
                                                          combined_without_intent_two["result"].values)
        sk_class_two_with_intent = accuracy_score(combined_with_intent_two["rel"].values,
                                                  combined_with_intent_two["result"].values)
        sk_class_two_without_intent = accuracy_score(combined_without_intent_two["rel"].values,
                                                     combined_without_intent_two["result"].values)

        # Label 3
        combined_with_intent_three = combined_with_intent[combined_with_intent["rel"] == 3]
        combined_without_intent_three = combined_without_intent[combined_without_intent["rel"] == 3]
        class_three_with_intent = accuracy_for_agreement(combined_with_intent_three["rel"].values,
                                                        combined_with_intent_three["result"].values)
        class_three_without_intent = accuracy_for_agreement(combined_without_intent_three["rel"].values,
                                                           combined_without_intent_three["result"].values)
        sk_class_three_with_intent = accuracy_score(combined_with_intent_three["rel"].values,
                                                    combined_with_intent_three["result"].values)
        sk_class_three_without_intent = accuracy_score(combined_without_intent_three["rel"].values,
                                                       combined_without_intent_three["result"].values)

        # Collapse positive relevance into a single value, making it a binary evaluation
        combined_with_intent["bin_rel"] = combined_with_intent.apply(
            lambda x: int(x["rel"] >= 1), axis=1)
        combined_with_intent["bin_result"] = combined_with_intent.apply(
            lambda x: int(x["result"] >= 1), axis=1)
        combined_without_intent["bin_rel"] = combined_without_intent.apply(
            lambda x: int(x["rel"] >= 1), axis=1)
        combined_without_intent["bin_result"] = combined_without_intent.apply(
            lambda x: int(x["result"] >= 1), axis=1)

        # Binary classification accuracy
        with_intent_report_bin = classification_report(combined_with_intent["bin_rel"].values,
                                                   combined_with_intent["bin_result"].values, labels=[0, 1])
        without_intent_report_bin = classification_report(combined_without_intent["bin_rel"].values,
                                                      combined_without_intent["bin_result"].values,
                                                      labels=[0, 1])

        # Accuracy for Agreement Assessment
        agree_i = accuracy_for_agreement(combined_with_intent["rel"].values, combined_with_intent["result"].values)
        agree_no_i = accuracy_for_agreement(combined_without_intent["rel"].values, combined_without_intent["result"].values)

        # Binary Accuracy for Agreement Assessment
        bin_agree_i = accuracy_for_agreement(combined_with_intent["bin_rel"].values, combined_with_intent["bin_result"].values)
        bin_agree_no_i = accuracy_for_agreement(combined_without_intent["bin_rel"].values,
                                            combined_without_intent["bin_result"].values)

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
            result_file.write("\n\nBINARY CLASSIFICATION ACCURACY WITH INTENT\n\n")
            result_file.write(with_intent_report_bin)
            result_file.write("\n\nBINARY CLASSIFICATION ACCURACY WITHOUT INTENT\n\n")
            result_file.write(without_intent_report_bin)
            result_file.write("\n\nCOHEN'S KAPPA WITH INTENT\n\n")
            result_file.write(f"Kappa:\t{cohen_i}")
            result_file.write("\n\nCOHEN'S KAPPA WITHOUT INTENT\n\n")
            result_file.write(f"Kappa:\t{cohen_no_i}")
            result_file.write("\n\nACCURACY FOR AGREEMENT WITH INTENT\n\n")
            result_file.write(f"Accuracy:\t{agree_i}")
            result_file.write("\n\nACCURACY FOR AGREEMENT WITHOUT INTENT\n\n")
            result_file.write(f"Accuracy:\t{agree_no_i}")
            result_file.write(f"\n\nBINARY ACCURACY FOR AGREEMENT WITH INTENT\n\n")
            result_file.write(f"Accuracy:\t{bin_agree_i}")
            result_file.write(f"\n\nBINARY ACCURACY FOR AGREEMENT WITHOUT INTENT\n\n")
            result_file.write(f"Accuracy:\t{bin_agree_no_i}")
            result_file.write(f"\n\nCLASS 0 ACCURACY WITH INTENT\n\n")
            result_file.write(f"\t{class_zero_with_intent}")
            result_file.write(f"\n\nCLASS 0 ACCURACY WITHOUT INTENT\n\n")
            result_file.write(f"\t{class_zero_without_intent}")
            result_file.write(f"\n\nCLASS 1 ACCURACY WITH INTENT\n\n")
            result_file.write(f"\t{class_one_with_intent}")
            result_file.write(f"\n\nCLASS 1 ACCURACY WITHOUT INTENT\n\n")
            result_file.write(f"\t{class_one_without_intent}")
            result_file.write(f"\n\nCLASS 2 ACCURACY WITH INTENT\n\n")
            result_file.write(f"\t{class_two_with_intent}")
            result_file.write(f"\n\nCLASS 2 ACCURACY WITHOUT INTENT\n\n")
            result_file.write(f"\t{class_two_without_intent}")
            result_file.write(f"\n\nCLASS 3 ACCURACY WITH INTENT\n\n")
            result_file.write(f"\t{class_three_with_intent}")
            result_file.write(f"\n\nCLASS 3 ACCURACY WITHOUT INTENT\n\n")
            result_file.write(f"\t{class_three_without_intent}")
            result_file.write(f"\n\nSCIKIT CLASS 0 ACCURACY WITH INTENT\n\n")
            result_file.write(f"\t{sk_class_zero_with_intent}")
            result_file.write(f"\n\nSCIKIT CLASS 0 ACCURACY WITHOUT INTENT\n\n")
            result_file.write(f"\t{sk_class_zero_without_intent}")
            result_file.write(f"\n\nSCIKIT CLASS 1 ACCURACY WITH INTENT\n\n")
            result_file.write(f"\t{sk_class_one_with_intent}")
            result_file.write(f"\n\nSCIKIT CLASS 1 ACCURACY WITHOUT INTENT\n\n")
            result_file.write(f"\t{sk_class_one_without_intent}")
            result_file.write(f"\n\nSCIKIT CLASS 2 ACCURACY WITH INTENT\n\n")
            result_file.write(f"\t{sk_class_two_with_intent}")
            result_file.write(f"\n\nSCIKIT CLASS 2 ACCURACY WITHOUT INTENT\n\n")
            result_file.write(f"\t{sk_class_two_without_intent}")
            result_file.write(f"\n\nSCIKIT CLASS 3 ACCURACY WITH INTENT\n\n")
            result_file.write(f"\t{sk_class_three_with_intent}")
            result_file.write(f"\n\nSCIKIT CLASS 3 ACCURACY WITHOUT INTENT\n\n")
            result_file.write(f"\t{sk_class_three_without_intent}")


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

if __name__ == "__main__":
    main()