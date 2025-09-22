import logging
import pandas as pd
from judging_the_intent import __version__
from judging_the_intent.db.schema import (
    Annotation,
    Config,
    Query,
    Triple,
)

LOGGER = logging.getLogger(__file__)

class Evaluator:
    def __init__(self, model: str, data_dir: str, dataset: str) -> None:
        self._model = model
        self._dataset = dataset
        self._data_dir = data_dir

    def _retrieve_database_annotations(self):
        """
        Internal method to retrieve annotations for a single dataset and model.

        :return: Tuple(pd.DataFrame, pd.DataFrame) where the first dataframe is intent-aware annotations, and the second
                 is not.
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

        # Get all Query objects for the current dataset
        dataset_queries = (
            Query.select()
            .where(Query.dataset_name == self._dataset)
            .alias("dataset_queries")
        )

        # Get all Triple objects that have ForeignKey relationships to the dataset Query objects, that have Intents
        dataset_triples_with_intent = (
            Triple.select()
            .where(Triple.intent.is_null(False))
            .join(dataset_queries, on=(Triple.query == dataset_queries.c.q_id))
            .join_from(Triple, Query)
        )

        # Get all Triple objects that have ForeignKey relationships to the dataset Query objects, that do not have Intents
        dataset_triples_without_intent = (
            Triple.select()
            .where(Triple.intent.is_null())
            .join(dataset_queries, on=(Triple.query == dataset_queries.c.q_id))
            .join_from(Triple, Query)
        )

        # Collect the related Annotation objects for the Triple entries, with and without intent
        model_annotations_with_intent = (
            Annotation.select(
                Annotation,
                Triple.query.alias("query_id"),
                Triple.intent.alias("intent_id"),
                Triple.document.alias("doc_id"),
            )
            .where(Annotation.result.in_([0, 1, 2, 3]))
            .join(Config, on=(Annotation.config == config.id))
            .join_from(Annotation, dataset_triples_with_intent,
                       on=(Annotation.triple == dataset_triples_with_intent.c.id))
            .join_from(Annotation, Triple)
        )

        model_annotations_without_intent = (
            Annotation.select(
                Annotation,
                Triple.query.alias("query_id"),
                Triple.intent.alias("intent_id"),
                Triple.document.alias("doc_id"),
            )
            .where(Annotation.result.in_([0, 1, 2, 3]))
            .join(Config, on=(Annotation.config == config.id))
            .join_from(Annotation, dataset_triples_without_intent,
                       on=(Annotation.triple == dataset_triples_without_intent.c.id))
            .join_from(Annotation, Triple)
        )

        with_intent = pd.DataFrame(model_annotations_with_intent.dicts())
        without_intent = pd.DataFrame(model_annotations_without_intent.dicts())
        LOGGER.info(
            f"Loaded {with_intent.shape[0]} LLM judgments with intent and {without_intent.shape[0]} without intent.")

        return with_intent, without_intent

    def run(self):
        raise NotImplemented("Define your own evaluation run method.")