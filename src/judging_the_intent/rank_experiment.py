import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from judging_the_intent.util.eval import Evaluator
from judging_the_intent.util.rank import rank, rank_correlation
from tqdm import tqdm


LOGGER = logging.getLogger(__file__)


class RankEvaluator(Evaluator):
    """
    Class to interface with ir_datasets_subsamples and PyTerrier to run a ranking experiment for human annotations,
    llm annotations, and intent-aware llm annotations.
    """
    def __init__(self, model: str, data_dir: str, dataset: str) -> None:
        super().__init__(model, data_dir, dataset)

    def run(self):
        """
        Run the evaluation

        Retrieves the Annotations for given Model and Dataset pair, then performs the evaluation
        """

        with_intent, without_intent = self._retrieve_database_annotations()

        # Make the fields match names and data types for the PyTerrier Experiment expectation
        with_intent = with_intent[["query_id", "intent_id", "doc_id", "result"]]
        with_intent.rename(columns={"result": "relevance", "query_id": "qid"}, inplace=True)
        with_intent = with_intent.dropna(subset=["relevance"])
        with_intent["relevance"] = with_intent["relevance"].apply(round)
        without_intent["relevance"] = with_intent["relevance"].astype("int64")

        without_intent = without_intent[["query_id", "intent_id", "doc_id", "result"]]
        without_intent.rename(columns={"result": "relevance", "query_id": "qid"}, inplace=True)
        without_intent = without_intent.dropna(subset=["relevance"])
        without_intent["relevance"] = without_intent["relevance"].astype("int64")

        return rank(self._dataset, self._data_dir, with_intent, without_intent)

    def corr(self):
        # nDCG correlation
        rank_correlation(self._model, self._dataset)

        # ERR correlation
        rank_correlation(self._model, self._dataset, "err")


def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--models", required=True, nargs="+", help="Ollama model identifiers.")
    ap.add_argument("-d", dest="datasets", required=True, nargs="+", help="Dataset identifiers")
    ap.add_argument("--data_dir", help="Directory containing the TREC Qrels files.",
                    default=str(Path(__file__).parent.parent.parent.joinpath("trec-web", "qrels")))
    ap.add_argument("-c", "--corr", dest="corr", action="store_true", help="Flag to indicate running the correlation analysis")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.corr:
        # Create output directory
        output_path = Path(__file__).parent.parent.parent.joinpath("trec-web", "rank-output")
        output_path.mkdir(exist_ok=True)

        pbar = tqdm(args.datasets, total=len(args.datasets), desc=">> Beginning PyTerrier ranking...\t")
        for dataset in pbar:
            pbar.set_description(f">> Running PyTerrier ranking for {dataset}:\t")
            for model in args.models:
                human_outcome, llm_outcome, llm_outcome_si = RankEvaluator(model, args.data_dir, dataset).run()
                human_outcome.to_csv(Path(output_path).joinpath(
                    f"{model.replace(':', '-')}-{dataset.replace('/', '-')}-human-gt.tsv"),
                    index=False, sep="\t")
                llm_outcome.to_csv(Path(output_path).joinpath(
                    f"{model.replace(':', '-')}-{dataset.replace('/', '-')}-llm-gt-intent.tsv"),
                    index=False, sep="\t")
                llm_outcome_si.to_csv(Path(output_path).joinpath(
                    f"{model.replace(':', '-')}-{dataset.replace('/', '-')}-llm-gt-no-intent.tsv"),
                    index=False, sep="\t")
    else:
        for dataset in args.datasets:
            for model in args.models:
                RankEvaluator(model, "", dataset).corr()


if __name__ == "__main__":
    main()