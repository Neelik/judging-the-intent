import logging
import pandas as pd
from judging_the_intent import __version__
from judging_the_intent.db.schema import (
    Annotation,
    Config,
    Query,
    Triple,
)
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report

LOGGER = logging.getLogger(__file__)

class Evaluator:
    def __init__(self, model: str, qrels_true_path: str, dataset: str,
                 target_type: str, intent_aware: bool, checkpointed_model: bool) -> None:
        self._model = model
        self._dataset = dataset
        self._qrels_true_path = qrels_true_path
        assert target_type in ["binary", "multi_num"]
        self._target_type = target_type
        self._intent_aware = intent_aware
        self._checkpointed_model = checkpointed_model

    def _retrieve_database_annotations(self):
        """
        Internal method to retrieve annotations for a single dataset and model.
        :return: pd.DataFrame
        """

        config, created = Config.get_or_create(
            model_name=self._model, version=__version__, with_intent=self._intent_aware, fine_tuned=self._checkpointed_model,
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

        if self._intent_aware:
            # Get all Triple objects that have ForeignKey relationships to the dataset Query objects, that have Intents
            triples = (
                Triple.select()
                .where(Triple.intent.is_null(False))
                .join(dataset_queries, on=(Triple.query == dataset_queries.c.q_id))
                .join_from(Triple, Query)
            )

        else:
            # Get all Triple objects that have ForeignKey relationships to the dataset Query objects, that do not have Intents
            triples = (
                Triple.select()
                .where(Triple.intent.is_null())
                .join(dataset_queries, on=(Triple.query == dataset_queries.c.q_id))
                .join_from(Triple, Query)
            )

        model_annotations_from_db = (
            Annotation.select(
                Annotation,
                Triple.query.alias("query_id"),
                Triple.intent.alias("intent_id"),
                Triple.document.alias("doc_id"),
            )
            .where(Annotation.result.in_([0, 1, 2, 3, 4]))
            .join(Config, on=(Annotation.config == config.id))
            .join_from(Annotation, triples,
                       on=(Annotation.triple == triples.c.id))
            .join_from(Annotation, Triple)
        )

        annotations = pd.DataFrame(model_annotations_from_db.dicts())
        # For some reason, there are duplicates of triples (which by definition should be unique), so manual de-duplication is needed
        annotations = annotations.drop_duplicates(subset=["triple"])

        LOGGER.info(
            f"\tLoaded {annotations.shape[0]} LLM judgments.")

        return annotations

    def evaluate(self):
        # Retrieve the Annotation entries from the database (predictions)
        model_annotations = self._retrieve_database_annotations()
        model_annotations = model_annotations[["query_id", "intent_id", "doc_id", "result"]]

        # Load the human judgments (ground truth) for comparison
        human_df = pd.read_csv(Path(self._qrels_true_path), sep="\t",
                               names=["query_id", "intent_id", "doc_id", "rel"])

        # Filter out any rows that have negative relevance scores
        human_df = human_df[human_df["rel"] >= 0].copy()
        LOGGER.info(f"\tLoaded {human_df.shape[0]} human judgments.")

        assert model_annotations.shape[0] == human_df.shape[0]

        human_df["rel"] = human_df["rel"].apply(lambda x: 1 if x >= 1 else 0)
        true_list = human_df["rel"].values.tolist()
        pred_list = model_annotations["result"].values.tolist()

        self._calculate_metrics(true_list, pred_list)

    def _calculate_metrics(self, true_list, pred_list):
        place = 3
        labels = [0, 1] if self._target_type == "binary" else [0, 1, 2, 3, 4]
        target_names = ["Relevant", "Not Relevant"] if self._target_type == "binary" else \
            ["Fully Meets", "Highly Meets", "Moderately Meets", "Slightly Meets", "Fails to Meet"]
        cohen_kappa = cohen_kappa_score(true_list, pred_list)
        accuracy = accuracy_score(true_list, pred_list)
        report = classification_report(true_list, pred_list,
                                       labels=labels,
                                       output_dict=True,
                                       digits=3,
                                       target_names=target_names)

        matrix = confusion_matrix(true_list, pred_list, labels=labels)

        # sanity check
        assert accuracy == report['accuracy']
        for idx, class_name in enumerate(target_names):
            assert report[class_name]["support"] == matrix[idx].sum()

        # print something that is not easy to store
        print("confusion_matrix:\n", matrix.T)
        print(classification_report(true_list, pred_list,
                                    labels=labels,
                                    digits=place,
                                    target_names=target_names))

        result_dict = {"cohen_kappa": round(cohen_kappa, place),
                       "accuracy": round(report['accuracy'], place),
                       "f1-score": round(report['macro avg']['f1-score'], place),
                       "precision": round(report['macro avg']['precision'], place),
                       "recall": round(report['macro avg']['recall'], place),
                       "num": report['macro avg']["support"]}

        print(result_dict)