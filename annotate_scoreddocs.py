from argparse import ArgumentParser
from collections import defaultdict

import ir_datasets
from tqdm import tqdm

from dna_prompt import OllamaTripleAnnotator

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset",
        help="ir-datasets identifier with scored documents",
        default="msmarco-passage-v2/trec-dl-2021/judged",
    )
    args = ap.parse_args()

    dataset = ir_datasets.load(args.dataset)
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    docs_store = dataset.docs_store()

    qid_to_docids = defaultdict(list)
    for scoreddoc in tqdm(dataset.scoreddocs_iter(), total=dataset.scoreddocs_count()):
        qid_to_docids[scoreddoc.query_id].append(scoreddoc.doc_id)

    def triple_generator():
        for q_id, doc_ids in tqdm(qid_to_docids.items()):
            query = queries[q_id]
            query_intents = [("i1", "todo: get intents here")]  # get from DL-MIA
            for intent_id, intent in query_intents:
                for doc_id in doc_ids:
                    yield (
                        (q_id, query),
                        (intent_id, intent),
                        (doc_id, docs_store.get(doc_id).text),
                    )

    annotator = OllamaTripleAnnotator("mistral", triple_generator())
    annotator.configure()
    for j in annotator.get_judgments():
        print(j)
