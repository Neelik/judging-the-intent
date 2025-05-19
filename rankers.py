from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from ir_datasets_subsample import register_subsamples
from scipy.stats import spearmanr
# import ir_datasets
import pyterrier as pt
import pandas as pd

from pathlib import Path


def get_index(dataset_id):
    index_dir = Path("/tmp/index/" + (dataset_id.replace('/', '-')))
    pt_dataset = pt.datasets.get_dataset("irds:" + dataset_id)

    if not index_dir.exists() or not (index_dir / "data.properties").exists():
        indexer = pt.IterDictIndexer(str(index_dir), overwrite=True, meta={"docno": 100, "text": 20480})
        indexer.index(pt_dataset.get_corpus_iter())

    return pt.IndexFactory.of(str(index_dir))


def main(dataset_id):
    pt_dataset = pt.datasets.get_dataset("irds:" + dataset_id)
    query_field = "title" if "misinfo" in dataset_id else "query"
    topics = pt_dataset.get_topics(query_field)
    index = get_index(dataset_id)

    # PyTerrier needs to use pre-tokenized queries
    tokeniser = pt.java.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
    topics["query"] = topics["query"].apply(lambda i: " ".join(tokeniser.getTokens(i)))
    qrels = pd.read_csv(Path(__file__).parent.joinpath("trec-web", "qrels",
                                                       f"{dataset_id.replace('/', '-')}-filtered-qrels.tsv"),
                        sep="\t", names=["qid", "iid", "doc_id", "relevance"])
    qrels["qid"] = qrels["qid"].astype(str)

    bm25 = pt.terrier.Retriever(index, wmodel="BM25")
    pl2 = pt.terrier.Retriever(index, wmodel="PL2")
    tf = pt.terrier.Retriever(index, wmodel="Tf")

    # Human QRels
    human_outcome = pt.Experiment(
        [bm25, pl2, tf],
        topics=topics,
        qrels=qrels,
        eval_metrics=["ndcg_cut.10"],
        names=["BM25", "PL2", "Tf"]
    )

    # LLM QRels with Intent
    llm_qrels = pd.read_csv(Path(__file__).parent.joinpath("trec-web", "qrels", "trec_web_2009_mistral.tsv"),
                        sep="\t", names=["qid", "iid", "doc_id", "relevance", "model"])
    llm_qrels["qid"] = llm_qrels["qid"].astype(str)
    llm_qrels["relevance"] = llm_qrels["relevance"].apply(lambda x: round(x))
    llm_outcome = pt.Experiment(
        [bm25, pl2, tf],
        topics=topics,
        qrels=llm_qrels,
        eval_metrics=["ndcg_cut.10"],
        names=["BM25", "PL2", "Tf"]
    )

    # LLM QRels without Intent
    llm_qrels_si = pd.read_csv(Path(__file__).parent.joinpath("trec-web", "qrels", "trec_web_2009_sans_intent_mistral.tsv"),
                        sep="\t", names=["qid", "iid", "doc_id", "relevance", "model"])
    llm_qrels_si["qid"] = llm_qrels_si["qid"].astype(str)
    llm_qrels_si["relevance"] = llm_qrels_si["relevance"].apply(lambda x: round(x))
    llm_outcome_si = pt.Experiment(
        [bm25, pl2, tf],
        topics=topics,
        qrels=llm_qrels_si,
        eval_metrics=["ndcg_cut.10"],
        names=["BM25", "PL2", "Tf"]
    )

    return human_outcome, llm_outcome, llm_outcome_si


def evaluate_systems(dataset, metric_name="ndcg_cut.10"):
    human_performance = pd.read_csv(
        Path().cwd().joinpath("results", "clueweb", f"{dataset.replace('/', '-')}_pt_experiments_human.csv"))
    llm_performance = pd.read_csv(
        Path().cwd().joinpath("results", "clueweb", f"{dataset.replace('/', '-')}_pt_experiments_llm.csv"))
    llm_si_performance = pd.read_csv(
        Path().cwd().joinpath("results", "clueweb", f"{dataset.replace('/', '-')}_pt_experiments_llm_si.csv"))

    print(set(human_performance[metric_name].values))
    print(set(llm_performance[metric_name].values))
    print(set(llm_si_performance[metric_name].values))

    print("INTENT-DRIVEN RANK CORRELATION")
    print(spearmanr(human_performance[metric_name].values, llm_performance[metric_name].values))

    print("INTENT-FREE RANK CORRELATION")
    print(spearmanr(human_performance[metric_name].values, llm_si_performance[metric_name].values))


if __name__ == "__main__":
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("--d", dest="DATASET", help="Name of ir_dataset to perform annotation on.")
    args = ap.parse_args()

    register_subsamples()
    # datasets = ["irds:msmarco-passage-v2/trec-dl-2021/judged", "irds:msmarco-passage-v2/trec-dl-2022/judged"]
    # datasets = ["corpus-subsamples/clueweb09/en/trec-web-2009", "corpus-subsamples/clueweb09/en/trec-web-2010",
    #             "corpus-subsamples/clueweb09/en/trec-web-2011", "corpus-subsamples/clueweb09/en/trec-web-2012",
    #             "corpus-subsamples/clueweb12/trec-web-2013", "corpus-subsamples/clueweb12/trec-web-2014",
    #             "corpus-subsamples/clueweb12/b13/trec-misinfo-2019"]


    results = main(args.DATASET)
    results[0].to_csv(Path().cwd().joinpath("results", "clueweb", f"{args.DATASET.replace('/', '-')}_pt_experiments_human.csv"),
                      index=False)
    results[1].to_csv(Path().cwd().joinpath("results", "clueweb", f"{args.DATASET.replace('/', '-')}_pt_experiments_llm.csv"),
           index=False)
    results[1].to_csv(
        Path().cwd().joinpath("results", "clueweb", f"{args.DATASET.replace('/', '-')}_pt_experiments_llm_si.csv"),
        index=False)

    evaluate_systems(args.DATASET)
