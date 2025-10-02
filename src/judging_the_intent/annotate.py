import json
import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from transformers import AutoTokenizer

import requests
from peewee import JOIN
from tqdm import tqdm

from judging_the_intent import __version__
from judging_the_intent.db.schema import (
    Annotation,
    Config,
    Document,
    Intent,
    Query,
    Triple,
)
from judging_the_intent.util.dna_prompt import build_prompt
from judging_the_intent.util.tokenizers import tokenizer_lookup
from judging_the_intent.util.parsers import Parser, Phi4Parser

LOGGER = logging.getLogger(__file__)
OLLAMA_API = f"{os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}/api"
HF_ACCESS_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")


class Annotator:
    """Wrapper class allowing for inference calls to Ollama models using the DNA prompt format.

    :param model: Name of the Ollama model to be used in inference.
    """

    def __init__(self, model: str, parser: Parser) -> None:
        self._model = model
        self._parser = parser
        self._dataset_name = None

    def set_dataset(self, dataset_name: str) -> None:
        self._dataset_name = dataset_name

    def run(self) -> None:
        """Run the annotation.

        Retrieves triples without annotations from the database, annotates them using the LLM,
        and writes the results back into the database.
        """
        config, created = Config.get_or_create(
            model_name=self._model, version=__version__
        )
        if created:
            LOGGER.info(
                "model %s (version %s) not found in DB, creating",
                self._model,
                __version__,
            )
        else:
            LOGGER.info("found model %s (version %s) in DB", self._model, __version__)

        # If there is a dataset_name set, then filter the unannotated triples specifically for that dataset
        if self._dataset_name:
            queries = (
                Query.select()
                .where(Query.dataset_name == self._dataset_name)
            )
        else:
            queries = Query.select()

        # select all triples except the ones that are already annotated
        # this includes annotation with errors
        unannotated_triples_cte = (
            Triple.select()
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
        count = unannotated_triples.count()
        LOGGER.info("%s triples left to annotate", count)

        for item in tqdm(unannotated_triples.dicts(), total=count):
            data = {
                "prompt": build_prompt(
                    item["query_text"], item["intent_text"], item["document_text"],
                    version="verbose"
                ),
                "model": self._model,
                "stream": False,
                "format": {
                    "type": "object",
                    "properties": {
                        "Relevance Score": {
                            "type": "integer"
                        },
                        "Explanation": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "Relevance Score",
                        "Explanation"
                    ]
                }
            }

            # Before passing to the API, check whether the built prompt will be truncated based on the defined context window
            context_length = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", 4096))
            model_id = tokenizer_lookup(self._model)
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_ACCESS_TOKEN)
            tokenized_prompt = tokenizer.encode(data["prompt"])
            prompt_length = len(tokenized_prompt)
            if prompt_length > context_length:
                LOGGER.warning(f"Triple {item['id']} exceeded context length {context_length}.")

            result, error, explanation = None, None, None
            try:
                api_response = requests.post(
                    url=f"{OLLAMA_API}/generate",
                    data=json.dumps(data),
                    headers={"Content-Type": "application/json"},
                )
                result, explanation = self._parser(json.loads(api_response.text)["response"])
            except Exception as e:
                LOGGER.error("error while annotating triple %s", item["id"])
                error = repr(e)
            finally:
                # this will add or update an annotation
                Annotation.insert(
                    triple=item["id"],
                    config=config.id,
                    result=result,
                    error=error,
                    truncated=True if prompt_length >= context_length else False,
                    explanation=explanation,
                ).on_conflict(
                    conflict_target=[Annotation.triple, Annotation.config],
                    preserve=[Annotation.triple, Annotation.config],
                    update={Annotation.result: result, Annotation.error: error, Annotation.explanation: explanation},
                ).execute()


def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--models", required=True, nargs="+", help="Ollama model identifiers."
    )
    ap.add_argument("--dataset", nargs=1, required=False, help="IR Datasets dataset identifier. Single dataset only.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    # available parsers in order of preference
    parsers = [Phi4Parser(), Parser()]

    for model in args.models:
        for parser in parsers:
            if parser.matches(model):
                LOGGER.info("processing %s using %s", model, parser.__class__.__name__)
                annotator = Annotator(model, parser)
                if args.dataset:
                    annotator.set_dataset(args.dataset)
                annotator.run()
                break


if __name__ == "__main__":
    main()
