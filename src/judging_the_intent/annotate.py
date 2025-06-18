import json
import logging
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

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
from judging_the_intent.util.parsers import Parser, Phi4Parser

LOGGER = logging.getLogger(__file__)
OLLAMA_API = f"{os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}/api"


class Annotator:
    """Wrapper class allowing for inference calls to Ollama models using the DNA prompt format.

    :param model: Name of the Ollama model to be used in inference.
    """

    def __init__(self, model: str, parser: Parser) -> None:
        self._model = model
        self._parser = parser

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

        # select all triples except the ones that are already annotated
        # this includes annotation with errors
        unannotated_triples_cte = (
            Triple.select()
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
                    item["query_text"], item["intent_text"], item["document_text"]
                ),
                "model": self._model,
                "stream": False,
            }

            result, error = None, None
            try:
                api_response = requests.post(
                    url=f"{OLLAMA_API}/generate",
                    data=json.dumps(data),
                    headers={"Content-Type": "application/json"},
                )
                result = self._parser(json.loads(api_response.text)["response"])
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
                ).on_conflict(
                    conflict_target=[Annotation.triple, Annotation.config],
                    preserve=[Annotation.triple, Annotation.config],
                    update={Annotation.result: result, Annotation.error: error},
                ).execute()


def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument(
        "--models", required=True, nargs="+", help="Ollama model identifiers."
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO)

    # available parsers in order of preference
    parsers = [Phi4Parser(), Parser()]

    for model in args.models:
        for parser in parsers:
            if parser.matches(model):
                LOGGER.info("processing %s using %s", model, parser.__class__.__name__)
                Annotator(model, parser).run()
                break


if __name__ == "__main__":
    main()
