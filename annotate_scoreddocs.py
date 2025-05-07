import csv
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from pathlib import Path

import ir_datasets
from tqdm import tqdm

from dna_prompt import OllamaTripleAnnotator


def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("MODEL", help="Ollama model")
    ap.add_argument("--out_file", default="out.tsv", help="Output file (.tsv)")
    args = ap.parse_args()

    print("reading intents...")
    with open(
        Path(__file__).parent / "DL-MIA" / "data" / "intent.tsv",
        encoding="utf-8",
        newline="",
    ) as fp:
        intents = {row[0]: row[1] for row in csv.reader(fp, delimiter="\t")}

    print("reading query-intent mappings...")
    q_id_to_i_d = defaultdict(set)
    with open(
        Path(__file__).parent / "DL-MIA" / "data" / "qid_iid_qrel.txt",
        encoding="utf-8",
        newline="",
    ) as fp:
        for row in csv.reader(fp, delimiter=" "):
            q_id_to_i_d[row[0]].add(row[1])

    print("reading document pools...")
    qid_to_docids = defaultdict(list)
    queries = {}
    for ds_name in (
        "msmarco-passage-v2/trec-dl-2021/judged",
        "msmarco-passage-v2/trec-dl-2022/judged",
    ):
        dataset = ir_datasets.load(ds_name)
        queries.update({q.query_id: q.text for q in dataset.queries_iter()})
        docs_store = dataset.docs_store()
        for qrel in tqdm(
            dataset.qrels_iter(), total=dataset.qrels_count()
        ):
            qid_to_docids[qrel.query_id].append(qrel.doc_id)

    def triple_generator():
        for q_id in tqdm(q_id_to_i_d):
            doc_ids = qid_to_docids[q_id]
            query = queries[q_id]
            for intent_id in q_id_to_i_d[q_id]:
                intent = intents[intent_id]
                for doc_id in doc_ids:
                    yield (
                        (q_id, query),
                        (intent_id, intent),
                        (doc_id, docs_store.get(doc_id).text),
                    )

    print("annotating...")
    annotator = OllamaTripleAnnotator(args.MODEL, triple_generator())
    annotator.configure()

    with open(args.out_file, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        for j in annotator.get_judgments():
            if "error" in j:
                print(
                    f"error for query {j['query_id']}, intent {j['intent_id']}, document {j['doc_id']}: {j['error']}"
                )
            else:
                writer.writerow(
                    [
                        j["query_id"],
                        j["intent_id"],
                        j["doc_id"],
                        j["relevance_score"],
                        args.MODEL,
                    ]
                )

    print(annotator.unload())

if __name__ == "__main__":
    main()
