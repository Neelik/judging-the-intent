import os
import ir_datasets
import pandas as pd
import torch
from ir_datasets_subsample import register_subsamples
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from judging_the_intent import __version__
from judging_the_intent.util.prompter import IntentGenerationPrompter
from judging_the_intent.db.schema import (
    Config,
    Dataset,
    Query,
    Triple,
    Document,
)
from peewee import JOIN
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from tqdm import tqdm
import logging
LOGGER = logging.getLogger(__name__)

MAX_DOC_LENGTH = 1024
HF_CACHE_DIR = os.environ.get("CACHE_DIR")
HF_ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN")


class IntentGenerator:
    def __init__(self, model: str, dataset: str, prompter: IntentGenerationPrompter) -> None:
        self._model_name = model
        self._dataset_identifier = dataset
        self._dataset = self._load_dataset()
        self._model = self._configure_model()
        self._tokenizer = self._configure_tokenizer()
        self._collection = self._build_collection()
        self._prompter = prompter

    def _configure_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            token=HF_ACCESS_TOKEN
        )
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        return model

    def _configure_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, padding_side="left",
                                                  cache_dir=HF_CACHE_DIR, token=HF_ACCESS_TOKEN)
        self._model.generation_config.pad_token_id = tokenizer.pad_token_id
        return tokenizer

    def _load_dataset(self):
        if "clueweb" in self._dataset_identifier:
            register_subsamples()
        return ir_datasets.load(self._dataset_identifier)

    def _build_collection(self):
        # Model name is 'human' here because we are loading the equivalent of the original Qrels from the database
        config, created = Config.get_or_create(
            model_name="human", version=__version__, with_intent=False,
            fine_tuned=False, prompt_style="human"
        )
        if created:
            LOGGER.info(
                "model %s (version %s) not found in DB, creating",
                "human",
                __version__,
            )
        else:
            LOGGER.info("found model %s (version %s) in DB", "human", __version__)

        dataset_queries = (
            Query.select()
            .where(Query.dataset_name_id.in_(Dataset.select().where(Dataset.name == self._dataset_identifier)))
            .alias("dataset_queries")
        )

        triples = (
            Triple.select(
                Triple.query_id,
                Triple.document_id,
                Query.text.alias("query_text"),
                Document.text.alias("document_text"),
            )
            .join(Query, JOIN.LEFT_OUTER, on=Triple.query_id == Query.id)
            .join(Document, JOIN.LEFT_OUTER, on=Triple.document_id == Document.d_id)
            .where((Triple.intent.is_null()) & (Triple.query_id.in_(dataset_queries)))
        )

        triples_dicts = triples.dicts()
        triples_frame = pd.DataFrame(triples_dicts)

        return triples_frame

        # collection = defaultdict(list)
        # if self._dataset.has_qrels():
        #     docs_store = self._dataset.docs_store()
        #     qrels = self._dataset.qrels_iter()
        #     for qrel in tqdm(qrels, total=self._dataset.qrels_count(), desc=">> Building collection..."):
        #         document = docs_store.get(qrel.doc_id)
        #         if qrel.query_id not in collection:
        #             collection[qrel.query_id] = []
        #         collection[qrel.query_id].append(document.text)
        #     return collection
        # else:
        #     LOGGER.warning("No queries available for this dataset")
        #     return []

    def run(self):
        qids = self._collection["query"].unique()
        qids_count = qids.shape[0]

        pipe = pipeline(
            task="text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
        )

        # TODO ClueWeb is massive documents, how many do we pass to the prompt? For now, three seems to work

        query_intents = defaultdict(list)
        system_role_message = {"role": "system", "content": "You are an intelligent system and your job is to predict the intention behind the user question given a list of documents."}
        for qid in tqdm(qids, total=qids_count, desc=">> Generating intents..."):
            filtered = self._collection[self._collection["query"] == qid]
            sampled_docs = filtered["document_text"].sample(n=3, random_state=42)
            documents = "\n".join([doc[:MAX_DOC_LENGTH] for doc in sampled_docs.values])
            filled_prompt = self._prompter.template.format(query=filtered["query_text"].values[0], documents=documents)
            user_role_message = {"role": "user", "content": filled_prompt}
            messages = [
                system_role_message,
                user_role_message,
            ]
            outputs = pipe(
                messages,
                max_new_tokens=256
            )

            decoded = outputs[0]["generated_text"][-1]
            try:
                intents = self._prompter.parser(decoded["content"])
                if qid not in query_intents:
                    query_intents[qid] = intents
                else:
                    LOGGER.info(f"\tQuery {qid} already has intents generated, ignoring.")
            except Exception as e:
                LOGGER.error(f"Parsing intents for {qid} failed. Generated output\n\n{decoded['content']}")

        # Write the intents to a file for manual inspection
        dataset_name_split = self._dataset_identifier.split("/")
        dataset_top_level_name = dataset_name_split[1]
        dataset_track = dataset_name_split[-1]
        output_path = Path(__file__).parent.parent.parent.joinpath("datasets", dataset_top_level_name,
                                                                   dataset_track, "intent")
        output_filename = f"{int(datetime.now().timestamp())}_{self._model_name.replace('/', '-')}_intents.tsv"
        intents_frame = pd.DataFrame.from_dict(query_intents.items())
        intents_frame.columns = ["query_id", "intent_text"]
        intents_frame = intents_frame.explode("intent_text")
        # Keep headers on this because the column names are the query ids
        intents_frame.to_csv(Path(output_path).joinpath(output_filename), index=False, sep="\t")
        LOGGER.info(f"\tSaved intents for {dataset_track} to {Path(output_path).joinpath(output_filename)}")


def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--model", required=True, default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model identifier. May require configuration of HF access token."
    )
    ap.add_argument("--datasets", nargs="+", required=False, help="IR Datasets dataset identifiers.")
    # ap.add_argument("--prompt_style", required=True, type=str, help="Define the prompt style to use in this intent generation run.")
    args = ap.parse_args()

    prompter = IntentGenerationPrompter(prompt_style="generate-intent")

    for dataset in args.datasets:
        LOGGER.info(f"\tGenerating intents for {dataset}...")
        generator = IntentGenerator(model=args.model, dataset=dataset, prompter=prompter)
        generator.run()

if __name__ == "__main__":
    main()