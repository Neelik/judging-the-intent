import pandas as pd
import logging
import os
import sys
import json
import requests
from pathlib import Path

LOGGER = logging.getLogger(__file__)

from judging_the_intent import __version__
from judging_the_intent.util.dna_prompt import build_prompt
from judging_the_intent.db.schema import (
    Annotation,
    Config,
    Document,
    Intent,
    Query,
    Triple,
)

OLLAMA_API = f"{os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}/api"

def main(doc_id, query_id, intent_id, model):
    logging.basicConfig(level=logging.INFO)

    config = Config.get(
        model_name=model, version=__version__
    )
    if config is not None:
        LOGGER.info("found model %s (version %s) in DB", model, __version__)
    else:
        LOGGER.error("could not find model %s", model)
        sys.exit(1)

    doc = Document.select().where(Document.d_id == doc_id)
    intent = Intent.select().where(Intent.i_id == intent_id)
    query = Query.select().where(Query.q_id == query_id)
    triple = Triple.select().where(Triple.intent.in_(intent)).where(Triple.query.in_(query)).where(Triple.document.in_(doc))
    annotation = Annotation.select().where(Annotation.triple.in_(triple))

    query = query.get_or_none()
    intent = intent.get_or_none()
    doc = doc.get_or_none()
    annotation = annotation.get_or_none()
    data_dir = Path(__file__).parent.parent.parent.parent.joinpath(
        "trec-web", "qrels", f"{query.dataset_name.replace('/', '-')}-filtered-qrels.tsv")
    dataset_df = pd.read_csv(data_dir, sep="\t", names=["query_id", "intent_id", "doc_id", "rel"])
    human_annotation = dataset_df[(dataset_df["query_id"] == query_id) & (dataset_df["intent_id"] == intent_id) & (dataset_df["doc_id"] == doc_id)]

    data = {
        "prompt": build_prompt(
            query.text, intent.text, doc.text
        ),
        "model": model,
        "stream": False,
    }

    try:
        api_response = requests.post(
            url=f"{OLLAMA_API}/generate",
            data=json.dumps(data),
            headers={"Content-Type": "application/json"},
        )
        result = json.loads(api_response.text)["response"]
        print(f"NEW:\t{json.loads(result)['Relevance Score']}\tOLD:\t{annotation.result}\tHUMAN:\t{human_annotation.rel.values[0]}\n\n")
        print(f"{json.loads(result)['Explanation']}")
    except Exception as e:
        print("Annotation failed.")

    # print(f"DOCUMENT\n{doc.text}\n\nQUERY\n{query.text}\n\nINTENT\n{intent.text}\n\nLLM Judgement: {annotation.result}\t\tHuman Judgement: {human_annotation.rel.values[0]}\n\n")


if __name__ == "__main__":
    running = True
    model = input("Please enter the Ollama model identifier: ")
    while running:
        doc_id = input("Enter document ID: ")
        query_id = int(input("Enter query ID: "))
        intent_id = int(input("Enter intent ID: "))

        try:
            main(doc_id, query_id, intent_id, model)
        except Exception:
            pass

        cont = input("Continue? (y/n): ")
        if cont == "n":
            running = False
