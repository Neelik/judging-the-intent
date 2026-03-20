import logging
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from judging_the_intent.util.eval import Evaluator
from judging_the_intent.util.rank import rank, rank_correlation
from tqdm import tqdm
from typing import Optional


LOGGER = logging.getLogger(__file__)


class RankEvaluator(Evaluator):
    """
    Class to interface with ir_datasets_subsamples and PyTerrier to run a ranking experiment for human annotations,
    llm annotations, and intent-aware llm annotations.
    """
    def __init__(self, model: str, dataset: str, target_type: str, intent_aware: bool,
                 checkpointed_model: bool, prompt_style: str, qrels_true_path: Optional[str] = None,
                 intent_source: str = "human") -> None:
        super().__init__(model=model, dataset=dataset, target_type=target_type, intent_aware=intent_aware,
                         checkpointed_model=checkpointed_model, prompt_style=prompt_style, intent_source=intent_source)

    def run(self):
        """
        Run the evaluation

        Retrieves the Annotations for given Model and Dataset pair, then performs the evaluation
        """
        human_config = self._get_config(human=True)
        llm_config = self._get_config(human=False)

        human_annotations_from_db = self._retrieve_database_annotations(human_config)
        llm_annotations_from_db = self._retrieve_database_annotations(llm_config)

        if self._intent_source == "generated":
            llm_annotations_from_db = llm_annotations_from_db.loc[
                llm_annotations_from_db.groupby(["query_id", "doc_id"])["result"].idxmax()]

        # Make the fields match names and data types for the PyTerrier Experiment expectation
        human_annotations_from_db = human_annotations_from_db[["query_id", "intent_id", "doc_id", "result"]]
        human_annotations_from_db.rename(columns={"result": "relevance", "query_id": "qid"}, inplace=True)
        # human_annotations_from_db = human_annotations_from_db.dropna(subset=["relevance"])
        human_annotations_from_db["relevance"] = human_annotations_from_db["relevance"].astype("int64")

        llm_annotations_from_db = llm_annotations_from_db[["query_id", "intent_id", "doc_id", "result"]]
        llm_annotations_from_db.rename(columns={"result": "relevance", "query_id": "qid"}, inplace=True)
        # llm_annotations_from_db = llm_annotations_from_db.dropna(subset=["relevance"])
        llm_annotations_from_db["relevance"] = llm_annotations_from_db["relevance"].astype("int64")

        # It can happen that intents are not generated for some query-doc pairs (YAY, LLM nonsense), so we drop the unmatched ones here
        model_pairs = [f"{a}#{b}" for a, b in
                       zip(llm_annotations_from_db["qid"].values, llm_annotations_from_db["doc_id"].values)]
        human_pairs = [f"{a}#{b}" for a, b in zip(human_annotations_from_db["qid"].values, human_annotations_from_db["doc_id"].values)]
        diff = set(human_pairs) - set(model_pairs)
        human_annotations_from_db = human_annotations_from_db.apply(lambda x: x if f"{x['qid']}#{x['doc_id']}" not in diff else None, axis=1)
        human_annotations_from_db.dropna(how="all", inplace=True)
        # For some reason the line with the lambda above converts relevance to a float, so after dropping null values, have to set the type back
        human_annotations_from_db["relevance"] = human_annotations_from_db["relevance"].astype("int64")


        assert llm_annotations_from_db.shape[0] == human_annotations_from_db.shape[0]

        return rank(self._dataset, human_annotations_from_db, llm_annotations_from_db, intent_source=self._intent_source if self._intent_source == "generated" else None)

    # def corr(self):
    #     # nDCG correlation
    #     rank_correlation(self._model, self._dataset)
    #
    #     # ERR correlation
    #     rank_correlation(self._model, self._dataset, "err")


def main():
    logging.basicConfig(level=logging.INFO, format='{levelname} - {asctime} - {module} - {message}', style="{",
                        datefmt="%Y-%m-%d %H:%M")
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--model", required=True, type=str, help="HuggingFace model identifier.")
    ap.add_argument("--datasets", required=True, nargs="+", help="Dataset identifiers")
    ap.add_argument("--intent_aware", action="store_true", default=False, help="Run the intent-aware evaluation.")
    ap.add_argument("--checkpointed_model", action="store_true",
                    help="Set to true if evaluation is on annotations from a fine-tuned model.")
    ap.add_argument("--prompt_style", required=True,
                    choices=("human", "human-intent", "binary", "binary-intent", "dna", "dna-intent"),
                    help="Prompt style identifier.")
    ap.add_argument("--intent_source", type=str, choices=("human", "generated"), help="Intent source identifier.")
    # ap.add_argument("-c", "--corr", dest="corr", action="store_true", help="Flag to indicate running the correlation analysis")
    args = ap.parse_args()

    # We've opted to use the choices argument to control the intent_source, but that makes a default impossible.
    # This code snippet acts as a functional default control
    if not args.intent_source:
        args.intent_source = "human"

    if args.intent_source == "generated" and not args.intent_aware:
        LOGGER.warning(f"Ranking with intent_source of {args.intent_source} is not supported without intent_aware. Setting intent_aware to True. Please add this flag in subsequent runs.")
        args.intent_aware = True

    checkpointed = False
    if args.checkpointed_model:
        checkpointed = True

    # if not args.corr:
    # Create output directory
    output_path = Path(__file__).parent.parent.parent.joinpath("datasets", "outputs", "rank")
    output_path.mkdir(exist_ok=True)

    pbar = tqdm(args.datasets, total=len(args.datasets), desc=">> Beginning PyTerrier ranking...\t")
    for dataset in pbar:
        pbar.set_description(f">> Running PyTerrier ranking for {dataset}:\t")
        suffix = f"{'-gen' if args.intent_source == 'generated' else ''}-gt{'-intent' if args.intent_aware else ''}.tsv"
        human_outcome, llm_outcome = RankEvaluator(
            model=args.model, dataset=dataset, target_type="binary", prompt_style=args.prompt_style,
            intent_aware=args.intent_aware, checkpointed_model=checkpointed, intent_source=args.intent_source).run()
        human_outcome.to_csv(Path(output_path).joinpath(
            f"{args.model.replace('/', '_')}-{dataset.replace('/', '-')}-human{suffix}"),
            index=False, sep="\t")
        llm_outcome.to_csv(Path(output_path).joinpath(
            f"{args.model.replace('/', '_')}-{dataset.replace('/', '-')}-llm{suffix}"),
            index=False, sep="\t")
    # else:
    #     for dataset in args.datasets:
    #         for model in args.models:
    #             RankEvaluator(model, "", dataset).corr()


if __name__ == "__main__":
    main()