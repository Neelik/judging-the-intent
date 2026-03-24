import os
import ir_datasets
import pandas as pd
import torch
import traceback
import logging
from abc import ABCMeta, abstractmethod
from datetime import datetime
from ir_datasets_subsample import register_subsamples
from pathlib import Path
from peewee import JOIN, ModelSelect, Model
from shutil import get_terminal_size
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from typing import NamedTuple

from judging_the_intent import __version__
from judging_the_intent.db import DATABASE
from judging_the_intent.db.schema import (
    Annotation,
    Config,
    Dataset,
    Query,
    Triple,
    Document,
    Intent,
)
from judging_the_intent.util.prompter import (
    GenerationPrompter,
    IntentGenerationPrompter,
    SubtopicGenerationPrompter
)

MAX_DOC_LENGTH = 1024
HF_CACHE_DIR = os.environ.get("CACHE_DIR")
HF_ACCESS_TOKEN = os.environ.get("HF_ACCESS_TOKEN")
LOGGER = logging.getLogger(__name__)


class Generator(metaclass=ABCMeta):
    def __init__(self, model: str, dataset: str, prompter: GenerationPrompter):
        self._model_name = model
        self._dataset_identifier = dataset
        self._dataset = self._load_dataset()
        self._model = self._configure_model()
        self._tokenizer = self._configure_tokenizer()
        self._collection = self._build_collection()
        self._prompter = prompter
        self._batch_size = 16

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
        model.eval()

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

    def _build_collection(self) -> ModelSelect:
        dataset_queries = (
            Query.select()
            .where(Query.dataset_name_id.in_(Dataset.select().where(Dataset.name == self._dataset_identifier)))
            .alias("dataset_queries")
        )

        triples = (
            Triple.select(
                Triple,
                Query.text.alias("query_text"),
                Document.text.alias("document_text"),
            )
            .join(Query, JOIN.LEFT_OUTER, on=Triple.query_id == Query.id)
            .join(Document, JOIN.LEFT_OUTER, on=Triple.document_id == Document.d_id)
            .where((Triple.intent.is_null()) & (Triple.query_id.in_(dataset_queries)))
        )

        return triples

    def generate(self, commit_to_db: bool = False, write_to_file: bool = False) -> None:
        count = self._collection.count(database=DATABASE)
        collection_tuples = self._collection.namedtuples()
        if isinstance(self._prompter, IntentGenerationPrompter):
            intent_source = "generated-intent"
        elif isinstance(self._prompter, SubtopicGenerationPrompter):
            intent_source = "generated-subtopic"
        else:
            raise NotImplementedError(f"Prompter {self._prompter.__class__.__name__} not implemented")

        # Configure batching of dataset
        it = range(0, count, self._batch_size)
        pipe = pipeline(
            task="text-generation",
            model=self._model,
            tokenizer=self._tokenizer
        )
        if isinstance(self._model.config.eos_token_id, list):
            pipe.tokenizer.pad_token_id = self._model.config.eos_token_id[0]  # llama 3 128001
        else:
            pipe.tokenizer.pad_token_id = self._model.config.eos_token_id  # llama 3 128001

        new_triples = list()
        for start_idx in tqdm(it):
            batch = slice(start_idx, start_idx + self._batch_size)
            batched_collection = collection_tuples[batch]
            samples = self._build_prompts(batched_collection)
            outputs = pipe(samples, batch_size=self._batch_size)
            decoded_outputs = [output[0]["generated_text"][-1] for output in outputs]

            for pos, item in enumerate(batched_collection):
                parsed_output = self._prompter.parse(decoded_outputs[pos]["content"])
                for intent in parsed_output:
                    if commit_to_db:
                        intent_db, triple_db = self._commit_to_database(intent, intent_source, pos, item)

                        new_triples.append(
                            {"triple_id": triple_db.id, "query_id": triple_db.query.id, "query_text": triple_db.query.text,
                             "intent_id": triple_db.intent.id, "intent_text": triple_db.intent.text,
                             "document_id": triple_db.document.id, "document_text": triple_db.document.text})
                    else:
                        # Build new_triples without the database ORM ids for the Triple and the Intent
                        try:
                            new_triples.append({"triple_id": None, "query_id": item.query, "query_text": item.query_text,
                                                "intent_id": None, "intent_text": intent, "document_id": item.document,
                                                "document_text": item.document_text})
                        except AttributeError as e:
                            LOGGER.error(f"{traceback.print_exc()}")

        if write_to_file:
            # Write the intents to a file for manual inspection
            dataset_name_split = self._dataset_identifier.split("/")
            dataset_top_level_name = dataset_name_split[1]
            dataset_track = dataset_name_split[-1]
            output_path = Path(__file__).parent.parent.parent.parent.joinpath("datasets", dataset_top_level_name,
                                                                              dataset_track, "intent")
            output_filename = (f"{int(datetime.now().timestamp())}_{self._model_name.replace('/', '-')}_"
                               f"{intent_source.replace('-', '_')}.tsv")
            intents_frame = pd.DataFrame(new_triples)
            # Keep headers on this because the column names are the query ids
            intents_frame.to_csv(Path(output_path).joinpath(output_filename), index=False, sep="\t")
            LOGGER.info(f"\tSaved intents for {dataset_track} to {Path(output_path).joinpath(output_filename)}")
        else:
            # Write it out to the console
            print('-' * get_terminal_size(fallback=(80, 24)).columns)
            for output in new_triples:
                print("\n".join(f"{k}\t{v}" for k, v in output.items()))
                print('-' * get_terminal_size(fallback=(80, 24)).columns)

    @staticmethod
    def _commit_to_database(generated_entity: str, intent_source: str, batch_pos: int,
                            batch_item: NamedTuple) -> tuple[Model, Model]:
        with DATABASE.atomic():
            # Create the Intent entry
            intent_db = Intent.create(i_id=f"gen_{batch_pos}", query=batch_item.query, text=generated_entity,
                                      source=intent_source)
            # Create the Triple entry
            triple = Triple.create(intent=intent_db, query_id=batch_item.query, document_id=batch_item.document)

            return intent_db, triple

    @abstractmethod
    def _build_prompts(self, batch: list) -> list:
        raise NotImplemented


class SubtopicGenerator(Generator):
    def __init__(self, model: str, dataset: str, prompter: GenerationPrompter):
        super().__init__(model=model, dataset=dataset, prompter=prompter)
        self._batch_size = 32

    def _build_prompts(self, batch: list) -> list:
        messages = []
        system_role_message = {"role": "system",
                               "content": "You are an intelligent system and your job is to predict the intention behind the user's question based only on the question itself."}
        for query in batch:
            filled_prompt = self._prompter.build_prompt(query=query.text)
            user_role_message = {"role": "user", "content": filled_prompt}
            query_doc_messages = [
                system_role_message,
                user_role_message
            ]
            messages.append(query_doc_messages)

        return messages


class IntentGenerator(Generator):
    def __init__(self, model: str, dataset: str, prompter: GenerationPrompter) -> None:
        super().__init__(model=model, dataset=dataset, prompter=prompter)

    def _build_prompts(self, batch: list) -> list:
        messages = []
        system_role_message = {"role": "system",
                               "content": "You are an intelligent system and your job is to predict the intention behind the user question given a list of documents."}
        for query_doc in batch:
            documents = [query_doc.document_text[i:i + MAX_DOC_LENGTH]
                        for i in range(0, len(query_doc.document_text), MAX_DOC_LENGTH)]
            filled_prompt = self._prompter.build_prompt(query=query_doc.query_text, documents=documents)
            user_role_message = {"role": "user", "content": filled_prompt}
            query_doc_messages = [
                system_role_message,
                user_role_message
            ]
            messages.append(query_doc_messages)

        return messages