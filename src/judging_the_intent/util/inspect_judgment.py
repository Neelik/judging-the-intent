import pandas as pd
import logging
import os
import sys
import json
import requests
from pathlib import Path
from tqdm import tqdm

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

# def main(doc_id, query_id, intent_id, model):
def main(dataset, model):
    logging.basicConfig(level=logging.INFO)

    config = Config.get(
        model_name=model, version=__version__
    )
    if config is not None:
        LOGGER.info("found model %s (version %s) in DB", model, __version__)
    else:
        LOGGER.error("could not find model %s", model)
        sys.exit(1)

    filtered_qrels_path = Path(__file__).parent.parent.parent.parent.joinpath(
        "trec-web", "qrels", f"{dataset.replace('/', '-')}-filtered-qrels.tsv")
    queries_path = Path(__file__).parent.parent.parent.parent.joinpath(
        "trec-web", f"{dataset.replace('/', '-')}-queries.tsv")
    qid_iid_intents_path = Path(__file__).parent.parent.parent.parent.joinpath(
        "trec-web", f"{dataset.replace('/', '-')}-qid-iid-intent.tsv")
    queries = pd.read_csv(queries_path, sep="\t", names=["qid", "query"])
    qid_iid_intents = pd.read_csv(qid_iid_intents_path, sep="\t", names=["qid", "intent_id", "intent"])
    filtered_qrels = pd.read_csv(filtered_qrels_path, sep="\t", names=["query_id", "intent_id", "doc_id", "rel"])
    filtered_qrels = filtered_qrels[filtered_qrels["rel"] >= 2]

    intent_reannotations = []
    without_intent_reannotations = []

    LOGGER.info(f"Found {filtered_qrels.shape[0]} documents with positive relevance scores.")
    for entry in tqdm(filtered_qrels.itertuples(index=False), total=filtered_qrels.shape[0]):
        doc = Document.select().where(Document.d_id == entry.doc_id)
        intent_text = qid_iid_intents[(qid_iid_intents["qid"] == entry.query_id) & (qid_iid_intents["intent_id"] == entry.intent_id)]
        query_text = queries[queries["qid"] == entry.query_id]
        intent = Intent.select().where(Intent.text == intent_text["intent"].values[0])
        query = Query.select().where(Query.text == query_text["query"].values[0])
        triple = Triple.select().where(Triple.intent.in_(intent)).where(Triple.query.in_(query)).where(Triple.document.in_(doc))
        annotation = Annotation.select().where(Annotation.triple.in_(triple))

        query = query.get_or_none()
        intent = intent.get_or_none()
        doc = doc.get_or_none()
        annotation = annotation.get_or_none()
        # human_annotation = dataset_df[(dataset_df["query_id"] == entry.query_id) & (dataset_df["intent_id"] == entry.intent_id) & (dataset_df["doc_id"] == entry.doc_id)]
        data = {
            "prompt": build_prompt(
                query.text, intent.text, doc.text, version="verbose"
            ),
            "model": model,
            "stream": False,
        }

        LOGGER.info(" >> Seeking Explanation with Intent")
        try:
            api_response = requests.post(
                url=f"{OLLAMA_API}/generate",
                data=json.dumps(data),
                headers={"Content-Type": "application/json"},
            )
            result = json.loads(api_response.text)["response"]
            # print(f"NEW:\t{json.loads(result)['Relevance Score']}\tOLD:\t{annotation.result}\tHUMAN:\t{human_annotation.rel.values[0]}\n\n")
            # print(f"{json.loads(result)['Explanation']}")
            intent_reannotations.append({
                "query": query.text,
                "intent": intent.text,
                "doc": doc.text,
                "annotation": json.loads(result)['Relevance Score'],
                "explanation": json.loads(result)['Explanation'],
                "human_judgment": entry.rel,
                "old_llm_judgment": annotation.result,
            })
        except Exception as e:
            print("Annotation failed.")
            error = repr(e)
            print(error)

        LOGGER.info(" >> Seeking Explanation without Intent")
        data["prompt"] = build_prompt(query.text, None, doc.text, version="verbose")
        try:
            api_response = requests.post(
                url=f"{OLLAMA_API}/generate",
                data=json.dumps(data),
                headers={"Content-Type": "application/json"},
            )
            result = json.loads(api_response.text)["response"]
            # print(f"NEW:\t{json.loads(result)['Relevance Score']}\tOLD:\t{annotation.result}\tHUMAN:\t{human_annotation.rel.values[0]}\n\n")
            # print(f"{json.loads(result)['Explanation']}")
            without_intent_reannotations.append({
                "query": query.text,
                "intent": intent.text,
                "doc": doc.text,
                "annotation": json.loads(result)['Relevance Score'],
                "explanation": json.loads(result)['Explanation'],
                "human_judgment": entry.rel,
                "old_llm_judgment": annotation.result,
            })
        except Exception as e:
            print("Annotation failed.")
            error = repr(e)
            print(error)

    intent_reannotations_df = pd.DataFrame(intent_reannotations)
    without_intent_reannotations_df = pd.DataFrame(without_intent_reannotations)

    intent_reannotations_df.to_csv(
        Path(__file__).parent.parent.parent.parent.joinpath("trec-web", f"inspection-intent-{dataset.replace('/', '-')}.csv"),
        index=False)
    without_intent_reannotations_df.to_csv(
        Path(__file__).parent.parent.parent.parent.joinpath("trec-web", f"inspection-no-intent-{dataset.replace('/', '-')}.csv"),
        index=False)
    # print(f"DOCUMENT\n{doc.text}\n\nQUERY\n{query.text}\n\nINTENT\n{intent.text}\n\nLLM Judgement: {annotation.result}\t\tHuman Judgement: {human_annotation.rel.values[0]}\n\n")


if __name__ == "__main__":
    running = True
    model = input("Please enter the Ollama model identifier: ")
    dataset = input("Please enter the dataset identifier: ")
    main(dataset, model)
    # while running:
        # doc_id = input("Enter document ID: ")
        # query_id = int(input("Enter query ID: "))
        # intent_id = int(input("Enter intent ID: "))

        # try:
        #     main(doc_id, query_id, intent_id, model)
        # except Exception:
        #     pass
        #
        # cont = input("Continue? (y/n): ")
        # if cont == "n":
        #     running = False
