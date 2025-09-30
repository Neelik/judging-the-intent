import logging
from ir_datasets_subsample import register_subsamples
from scipy.stats import spearmanr
# from pyterrier_t5 import MonoT5ReRanker
import pyterrier as pt
import pandas as pd

from pathlib import Path
register_subsamples()


LOGGER = logging.getLogger(__file__)


def get_index(dataset_id):
    index_dir = Path("/tmp/index/" + (dataset_id.replace('/', '-')))
    pt_dataset = pt.datasets.get_dataset("irds:" + dataset_id)

    if not index_dir.exists() or not (index_dir / "data.properties").exists():
        indexer = pt.IterDictIndexer(str(index_dir), overwrite=True, meta={"docno": 100, "text": 20480})
        indexer.index(pt_dataset.get_corpus_iter())

    return pt.IndexFactory.of(str(index_dir))


def rank(dataset_id: str, qrels_directory: str, llm_with_intent: pd.DataFrame, llm_without_intent: pd.DataFrame):
    pt_dataset = pt.datasets.get_dataset("irds:" + dataset_id)
    query_field = "title" if "misinfo" in dataset_id else "query"
    topics = pt_dataset.get_topics(query_field)
    index = get_index(dataset_id)

    # PyTerrier needs to use pre-tokenized queries
    tokeniser = pt.java.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
    topics["query"] = topics["query"].apply(lambda i: " ".join(tokeniser.getTokens(i)))
    qrels = pd.read_csv(Path(qrels_directory).joinpath(f"{dataset_id.replace('/', '-')}-filtered-qrels.tsv"),
                        sep="\t", names=["qid", "iid", "doc_id", "relevance"])
    qrels["qid"] = qrels["qid"].astype(str)

    bm25 = pt.terrier.Retriever(index, wmodel="BM25")
    pl2 = pt.terrier.Retriever(index, wmodel="PL2")
    # monoT5 = MonoT5ReRanker()
    # monoT5 = bm25 >> monoT5

    # Human QRels
    human_outcome = pt.Experiment(
        [bm25, pl2],
        topics=topics,
        qrels=qrels,
        eval_metrics=["ndcg_cut.10", "recip_rank", "official"],
        names=["BM25", "PL2"],
    )

    # LLM QRels with Intent
    llm_outcome = pt.Experiment(
        [bm25, pl2],
        topics=topics,
        qrels=llm_with_intent,
        eval_metrics=["ndcg_cut.10", "recip_rank", "official"],
        names=["BM25", "PL2"]
    )

    # LLM QRels without Intent
    llm_outcome_si = pt.Experiment(
        [bm25, pl2],
        topics=topics,
        qrels=llm_without_intent,
        eval_metrics=["ndcg_cut.10", "recip_rank", "official"],
        names=["BM25", "PL2"]
    )

    return human_outcome, llm_outcome, llm_outcome_si


def rank_correlation(model, dataset, metric_name="ndcg_cut.10"):
    rank_output_path = Path(__file__).parent.parent.parent.parent.joinpath("trec-web", "rank-output")
    human_performance = pd.read_csv(Path(rank_output_path).joinpath(
        f"{model.replace(':', '-')}-{dataset.replace('/', '-')}-human-gt.tsv"),
        sep="\t", names=["ranker", metric_name])
    llm_performance = pd.read_csv(Path(rank_output_path).joinpath(
        f"{model.replace(':', '-')}-{dataset.replace('/', '-')}-llm-gt-intent.tsv"),
        sep="\t", names=["ranker", metric_name])
    llm_performance_si = pd.read_csv(Path(rank_output_path).joinpath(
        f"{model.replace(':', '-')}-{dataset.replace('/', '-')}-llm-gt-no-intent.tsv"),
        sep="\t", names=["ranker", metric_name])

    LOGGER.info(f"INTENT-DRIVEN RANK CORRELATION - {metric_name.capitalize()}")
    print(spearmanr(human_performance[metric_name].values, llm_performance[metric_name].values))

    LOGGER.info(f"INTENT-FREE RANK CORRELATION - {metric_name.capitalize()}")
    print(spearmanr(human_performance[metric_name].values, llm_performance_si[metric_name].values))
