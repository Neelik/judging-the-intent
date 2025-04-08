import csv
from collections import defaultdict
from pathlib import Path

import ir_datasets
from tqdm import tqdm

from dna_prompt import OllamaTripleAnnotator

if __name__ == "__main__":
    print("reading intents...")
    with open(
        Path.cwd() / "DL-MIA" / "data" / "intent.tsv", encoding="utf-8", newline=""
    ) as fp:
        intents = {row[0]: row[1] for row in csv.reader(fp, delimiter="\t")}

    print("reading query-intent mappings...")
    q_id_to_i_d = defaultdict(set)
    with open(
        Path.cwd() / "DL-MIA" / "data" / "qid_iid_qrel.txt",
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
        for scoreddoc in tqdm(
            dataset.scoreddocs_iter(), total=dataset.scoreddocs_count()
        ):
            qid_to_docids[scoreddoc.query_id].append(scoreddoc.doc_id)

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
    annotator = OllamaTripleAnnotator("mistral", triple_generator())
    annotator.configure()
    for j in annotator.get_judgments():
        print(j)
