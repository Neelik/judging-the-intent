import logging
import os
import peewee
import torch
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from peewee import JOIN, EXCLUDED
from tqdm import tqdm
from operator import or_
from functools import reduce

from judging_the_intent import __version__
from judging_the_intent.util.prompter import Prompter
from judging_the_intent.db.schema import (
    Annotation,
    Config,
    Document,
    Intent,
    Query,
    Triple,
)

LOGGER = logging.getLogger(__file__)
HF_ACCESS_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
HF_CACHE_DIR = os.environ.get("CACHE_DIR")


class Annotator:
    """Wrapper class allowing for inference calls to HuggingFace models using the DNA or Binary prompt formats.

    :param model: Name of the HuggingFace model to be used in inference.
    """

    def __init__(self, model: str, prompter: Prompter, batch_size: int, max_input_length: int, max_doc_length: int) -> None:
        self._model_name = model
        self._model = self._configure_model()
        self._tokenizer = self._configure_tokenizer()
        self._prompter = prompter
        self._datasets = None
        self._batch_size = batch_size
        self._max_input_length = max_input_length
        self._max_doc_length = max_doc_length
        self._checkpoint_loaded = False
        LOGGER.info(f"\tAnnotator initialized with {model}.")

    def load_checkpoint(self, checkpoint_path: str):
        self._model = PeftModel.from_pretrained(self._model, checkpoint_path)
        self._checkpoint_loaded = True
        LOGGER.info(f"\tCheckpoint for {self._model_name} loaded.")

    def set_dataset(self, dataset_names: list) -> None:
        self._datasets = dataset_names

    def _build_prompts(self, batch: list) -> list:
        """
        Method to take in a batch of query-doc pairs or query-intent-triples and build prompts prepared for the LLM
        :param batch: List of dicts containing the representations of the retrieved Triple objects (and related joins)
                      from the database.
        """
        if "intent" in self._prompter.prompt_style:

            samples = [self._prompter.template.format(demonstrations="", question=utd["query_text"],
                                                      intent=utd["intent_text"],
                                                      passage=utd["document_text"][:self._max_doc_length])
                       for utd in batch]
        else:
            samples = [self._prompter.template.format(demonstrations="", question=utd["query_text"],
                                                      passage=utd["document_text"][:self._max_doc_length])
                       for utd in batch]
        return samples


    def _configure_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, padding_side="left",
                                                  cache_dir=HF_CACHE_DIR, token=HF_ACCESS_TOKEN)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def _configure_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=HF_CACHE_DIR,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        )

        model.config.torch_dtype = torch.bfloat16
        model.config.pad_token_id = model.config.eos_token_id
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        if isinstance(model.generation_config.eos_token_id, list):
            model.generation_config.pad_token_id = model.generation_config.eos_token_id[0]  # llama 3 128001
        else:
            model.generation_config.pad_token_id = model.generation_config.eos_token_id  # llama 3 128001

        model.eval()
        return model

    def _load_unannotated_triples(self, config: Config) -> peewee.ModelSelect:
        # If there are datasets specified, then filter the unannotated triples specifically for those datasets
        if self._datasets:
            # Construct contains queries for each dataset to deal with overlaps
            conditions = []
            for dataset in self._datasets:
                col = getattr(Query, "dataset_name")
                conditions.append(col.contains(dataset))

            queries = (
                Query.select()
                .where((Query.dataset_name.in_(self._datasets)) | (reduce(or_, conditions)))
            )
        else:
            queries = Query.select()

        if "intent" in  self._prompter.prompt_style:
            # select all triples except the ones that are already annotated
            # this includes annotation with errors
            unannotated_triples_cte = (
                Triple.select()
                .where(Triple.intent.is_null(False))
                .where(Triple.query.in_(queries))
                .except_(
                    Triple.select()
                    .join(Annotation)
                    .join(Config)
                    .where(Config.id == config.id)
                    .where(Annotation.result.is_null(False))
                )
                .cte("unannotated_triples")
            )
        else:
            # select all triples except the ones that are already annotated
            # this includes annotation with errors
            unannotated_triples_cte = (
                Triple.select()
                .where(Triple.intent.is_null())
                .where(Triple.query.in_(queries))
                .except_(
                    Triple.select()
                    .join(Annotation)
                    .join(Config)
                    .where(Config.id == config.id)
                    .where(Annotation.result.is_null(False))
                )
                .cte("unannotated_triples")
            )

        # take the triples above and join them with query, intent, document texts
        unannotated_triples = (
            unannotated_triples_cte.select_from(
                unannotated_triples_cte.c.query_id,
                unannotated_triples_cte.c.intent_id,
                unannotated_triples_cte.c.document_id,
                unannotated_triples_cte.c.id,
                Query.text.alias("query_text"),
                Intent.text.alias("intent_text"),
                Document.text.alias("document_text"),
            )
            .join(Query, on=unannotated_triples_cte.c.query_id == Query.q_id)
            .join(
                Intent,
                JOIN.LEFT_OUTER,
                on=unannotated_triples_cte.c.intent_id == Intent.id,
            )
            .join(Document, on=unannotated_triples_cte.c.document_id == Document.d_id)
        )
        return unannotated_triples

    def run(self) -> None:
        """Run the annotation.

        Retrieves triples without annotations from the database, annotates them using the LLM,
        and writes the results back into the database.
        """
        config, created = Config.get_or_create(
            model_name=self._model_name, version=__version__, fine_tuned=self._checkpoint_loaded,
            with_intent=True if "intent" in self._prompter.prompt_style else False
        )
        if created:
            LOGGER.info(
                "\tmodel %s (version %s) not found in DB, creating",
                self._model_name,
                __version__,
            )
        else:
            LOGGER.info("\tfound model %s (version %s) in DB", self._model_name, __version__)

        unannotated_triples = self._load_unannotated_triples(config)
        count = unannotated_triples.count()
        unannotated_triples_dicts = unannotated_triples.dicts()
        LOGGER.info("\t%s triples left to annotate", count)

        # Configure batching of dataset
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        it = range(0, count, self._batch_size)

        for start_idx in tqdm(it):
            batch = slice(start_idx, start_idx + self._batch_size)
            batched_triples = unannotated_triples_dicts[batch]
            samples = self._build_prompts(batched_triples)

            # padding=True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence is provided).
            encoded = self._tokenizer(samples, padding=True, truncation=True,
                                      max_length=self._max_input_length, return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.inference_mode():
                predictions = self._model.generate(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    max_new_tokens=4,
                )

            predictions = self._tokenizer.batch_decode(predictions, skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True)

            annotations_for_db = []
            for pos, item in enumerate(samples):
                prediction = predictions[pos].split(self._prompter.splitter)[-1].strip()
                annotation_for_db = {
                    "triple": batched_triples[pos]["id"],
                    "config": config.id,
                    "result": self._prompter.parser(prediction),
                    "error": None,
                    "truncated": True, # all of them are truncated to 1400 characters in the document
                    "explanation": None # Explanations not implemented yet
                }
                annotations_for_db.append(annotation_for_db)

            Annotation.insert_many(annotations_for_db).on_conflict(
                conflict_target=[Annotation.triple, Annotation.config],
                preserve=[Annotation.triple, Annotation.config],
                update={Annotation.result: EXCLUDED.result, Annotation.error: EXCLUDED.error, Annotation.explanation: EXCLUDED.explanation},
            ).execute()


def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--model", required=True,
        help="HuggingFace model identifier. May require configuration of HF access token."
    )
    ap.add_argument("--datasets", nargs="+", required=False, help="IR Datasets dataset identifiers.")
    ap.add_argument("--prompt_style", required=True, type=str,
                    help="Define the prompt style to use in this annotation run.")
    ap.add_argument("--checkpoint_path", type=str, help="Path to checkpoint directory.", default=None)
    ap.add_argument("--batch_size", type=int, help="Batch size.", default=16)
    ap.add_argument("--max_input_length", type=int, help="Max input length.", default=2048)
    # Default is set to avoid OOM on GPU
    ap.add_argument("--max_doc_length", type=int, help="Max document length.", default=1400)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='{levelname} - {asctime} - {module} - {message}', style="{",
                        datefmt="%Y-%m-%d %H:%M")

    # Define the Prompter to be attached to the Annotator
    prompter = Prompter(args.prompt_style)

    LOGGER.info(f"\nInitializing annotation run with config:\n\tMODEL:\t{args.model}\n\tCHECKPOINT:\t"
                f"{('true' if args.checkpoint_path else 'false')}\n\tPROMPT:\t{args.prompt_style}")
    annotator = Annotator(args.model, prompter, args.batch_size, args.max_input_length, args.max_doc_length)
    if args.checkpoint_path:
        annotator.load_checkpoint(args.checkpoint_path)
    if args.datasets:
        annotator.set_dataset(args.datasets)
    annotator.run()


if __name__ == "__main__":
    main()
