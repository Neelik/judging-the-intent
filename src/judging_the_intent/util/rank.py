import logging
import os
from typing import Optional, TypeVar
from ir_datasets_subsample import register_subsamples
from scipy.stats import kendalltau, spearmanr
from pyterrier_t5 import MonoT5ReRanker
from rerank import LLMReRanker
import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier_dr import FlexIndex, TctColBert
import pandas as pd
from sentence_transformers import CrossEncoder

from pathlib import Path
register_subsamples()

LOGGER = logging.getLogger(__file__)
TPTTransformer = TypeVar("TPTTransformer", bound=pt.Transformer)


def get_index(dataset_id: str, dense: bool = False, model: Optional[TPTTransformer] = None):
    """
        Function to load a dense index for dense retrieval. In the case that the index is not created, creates one

        :param dataset_id: ir_dataset identifier for the dataset to index
        :param dense: If true, creates a dense index
        :param model: Option PyTerrier Transformer based model to use in creating a dense index
        :return IndexFactory: PyTerrier IndexFactory
    """

    index_dir = Path(f"{os.environ.get('PYTERRIER_INDEX_DIR')}{dataset_id.replace('/', '-')}")
    if dense:
        index_dir = Path(f"{index_dir}.flex")
    pt_dataset = pt.datasets.get_dataset("irds:" + dataset_id)

    if not index_dir.exists() or (len(list(index_dir.glob("**/*.properties"))) == 0):
        if dense:
            # Check for the pt_meta.json file
            if not Path(index_dir).joinpath("pt_meta.json").exists():
                LOGGER.info(f"\tCreating dense flex index for {dataset_id}")
                indexer = FlexIndex(str(index_dir))
                index_pipeline = model >> indexer
                index_pipeline.index(pt_dataset.get_corpus_iter())
        else:
            LOGGER.info(f"\tCreating index for {dataset_id}")
            indexer = pt.IterDictIndexer(str(index_dir), overwrite=True, meta={"docno": 100, "text": 20480})
            indexer.index(pt_dataset.get_corpus_iter())

    LOGGER.info(f"\tIndex for {dataset_id} loaded from {index_dir}")
    if dense:
        return FlexIndex(str(index_dir))
    else:
        return pt.IndexFactory.of(str(index_dir), memory=True)


def rank(dataset_id: str, human_annotations: pd.DataFrame, llm_annotations: pd.DataFrame, intent_aware: bool,
         intent_source: Optional[str] = None):
    pt_dataset = pt.datasets.get_dataset("irds:" + dataset_id)
    if "misinfo" in dataset_id:
        query_field = "title"
    elif "msmarco" in dataset_id:
        query_field = "text"
    else:
        query_field = "query"
    topics = pt_dataset.get_topics(query_field)
    index = get_index(dataset_id)

    # PyTerrier needs to use pre-tokenized queries
    tokeniser = pt.java.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
    topics["query"] = topics["query"].apply(lambda i: " ".join(tokeniser.getTokens(i)))
    human_annotations["qid"] = human_annotations["qid"].astype(str)
    llm_annotations["qid"] = llm_annotations["qid"].astype(str)

    # Lexical rankers, and baseline
    bm25 = pt.terrier.Retriever(index, wmodel="BM25")
    baseline_bm25 = bm25 % 100

    # Transformer reranker
    mono_t5 = MonoT5ReRanker(batch_size=32)
    mono_t5 = bm25 % 100 >> pt.text.get_text(pt_dataset, "text") >> mono_t5

    # Dense retrieval ranker
    tct_colbert = TctColBert("castorini/tct_colbert-v2-hnp-msmarco-r2")
    tct_colber_index = get_index(dataset_id, dense=True, model=tct_colbert)
    tct_colbert_retriever = tct_colbert >> tct_colber_index.np_retriever() % 100

    # Cross Encoder ranker
    # Max length is set to 512 as that's the longest the model can take
    cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L12-v2", max_length=512)
    def _cross_encoder_apply(df: pd.DataFrame):
        return cross_model.predict(list(zip(df["query"].values, df["text"].values)), show_progress_bar=False)

    cross_encoder_doc_scorer = pt.apply.doc_score(_cross_encoder_apply, batch_size=32)
    cross_encoder = bm25 % 100 >> pt.text.get_text(pt_dataset, "text") >> cross_encoder_doc_scorer

    # LLM ranker
    llm_reranker = LLMReRanker("castorini/rank_vicuna_7b_v1_fp16", top_k_candidates=10)
    llm_reranker = bm25 % 10 >> pt.text.get_text(pt_dataset, "text") >> llm_reranker

    # Experiment variables for PyTerrier pipeline
    retrieval_systems = [baseline_bm25, tct_colbert_retriever, mono_t5, cross_encoder, llm_reranker]
    system_names = ["BM25", "TCTColbert", "MonoT5", "CrossEncoder", "RankVicuna"]
    # retrieval_systems = [cross_encoder]
    # system_names = ["CrossEncoder"]
    metrics = ["ndcg_cut.10", "ndcg_cut.20", "recip_rank", "P_20", "map"]
    baseline = baseline_bm25.transform(pt_dataset.get_topics())

    # Configure save pathways
    dataset_name_split = dataset_id.split("/")
    dataset_track = dataset_name_split[-1]
    save_dir = Path(__file__).parent.parent.parent.parent.joinpath("datasets", "outputs", dataset_track)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup distinct names properly depending on generated intent type
    if intent_source == "generated-intent":
        human_names = [f"hum_gen_intent_{s}" for s in system_names]
        llm_names = [f"llm_gen_intent_{s}" for s in system_names]
    elif intent_source == "generated-subtopic":
        human_names = [f"hum_gen_subtopic_{s}" for s in system_names]
        llm_names = [f"llm_gen_subtopic_{s}" for s in system_names]
    else:
        human_names = [f"hum_{'intent_' if intent_aware else ''}{s}" for s in system_names]
        llm_names = [f"llm_{'intent_' if intent_aware else ''}{s}" for s in system_names]

    # Human QRels
    LOGGER.info(f"\tRunning experiment for human annotations of {dataset_id}")
    human_outcome = pt.Experiment(
        retr_systems=retrieval_systems,
        topics=topics,
        qrels=human_annotations,
        eval_metrics=metrics + [pta.RBO(baseline, p=0.9)],
        names=human_names,
        save_format=(pd.read_csv, pd.DataFrame.to_csv),
        save_dir=str(save_dir),
        save_mode="overwrite"
    )

    # LLM QRels with Intent
    LOGGER.info(f"\tRunning experiment for LLM annotations of {dataset_id}")
    llm_outcome = pt.Experiment(
        retr_systems=retrieval_systems,
        topics=topics,
        qrels=llm_annotations,
        eval_metrics=metrics + [pta.RBO(baseline, p=0.9)],
        names=llm_names,
        save_format=(pd.read_csv, pd.DataFrame.to_csv),
        save_dir=str(save_dir),
        save_mode="overwrite"
    )

    return human_outcome, llm_outcome


def rank_correlation(model: str, dataset: str, metric_name: str = "ndcg_cut.10", intent_source: str = "human",
                     intent_aware: bool = False) -> dict:
    rank_output_path = Path(__file__).parent.parent.parent.parent.joinpath("datasets", "outputs", "rank")
    suffix = (f"{'-gen' if 'generated' in intent_source else ''}{'-subtopic' if 'subtopic' in intent_source else ''}"
              f"-gt{'-intent' if intent_aware else ''}.tsv")

    human_performance = pd.read_csv(Path(rank_output_path).joinpath(
        f"{model.replace('/', '_')}-{dataset.replace('/', '-')}-human{suffix}"),
        sep="\t", names=["ranker", metric_name])

    llm_performance = pd.read_csv(Path(rank_output_path).joinpath(
        f"{model.replace('/', '_')}-{dataset.replace('/', '-')}-llm{suffix}"),
        sep="\t", names=["ranker", metric_name])
    # llm_performance_si = pd.read_csv(Path(rank_output_path).joinpath(
    #     f"{model.replace(':', '-')}-{dataset.replace('/', '-')}-llm-gt-no-intent.tsv"),
    #     sep="\t", names=["ranker", metric_name])
    k_tau, k_tau_p = kendalltau(human_performance[metric_name].values, llm_performance[metric_name].values)
    spearman_rho, spearman_rho_p = spearmanr(human_performance[metric_name].values, llm_performance[metric_name].values)

    results = {
        "model": model,
        "dataset": dataset,
        "k_tau": k_tau,
        "k_tau_p": k_tau_p,
        "spearman_rho": spearman_rho,
        "spearman_rho_p": spearman_rho_p
    }

    return results

    # LOGGER.info(f"INTENT-FREE RANK CORRELATION - {metric_name.capitalize()}")
    # print(kendalltau(human_performance[metric_name].values, llm_performance_si[metric_name].values))
