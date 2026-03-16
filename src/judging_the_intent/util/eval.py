import logging
import ir_datasets
from ir_datasets_subsample import register_subsamples
import pandas as pd
from judging_the_intent import __version__
from judging_the_intent.db.schema import (
    Annotation,
    Config,
    Dataset,
    Query,
    Triple,
    Intent,
)
from pathlib import Path
from typing import Optional
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report

LOGGER = logging.getLogger(__file__)

class Evaluator:
    def __init__(self, model: str, dataset: str, target_type: str, intent_aware: bool,
                 checkpointed_model: bool, prompt_style: str, qrels_true_path: Optional[str] = None,
                 intent_source: str = "human") -> None:
        self._model = model
        self._dataset = dataset
        self._qrels_true_path = qrels_true_path
        assert target_type in ["binary", "multi_num"]
        self._target_type = target_type
        self._intent_aware = intent_aware
        self._checkpointed_model = checkpointed_model
        self._prompt_style = prompt_style
        self._intent_source = intent_source

    def _get_config(self, human: bool = False) -> Config:
        """
            Internal method to retrieve the Config entity that will drive the retrieval of Annotations

            :param human: Flag to indicate whether to set the model to human for ground truth retrieval.
        """
        if human:
            fine_tuned = False
            model_name = "human"
            if self._intent_aware:
                prompt_style = "human-intent"
                intent_aware = False
            else:
                prompt_style = "human"
                intent_aware = self._intent_aware
        else:
            model_name = self._model
            prompt_style = self._prompt_style
            fine_tuned = self._checkpointed_model
            intent_aware = self._intent_aware

        config, created = Config.get_or_create(
            model_name=model_name, version=__version__, with_intent=intent_aware,
            fine_tuned=fine_tuned, prompt_style=prompt_style
        )
        if created:
            LOGGER.info(
                "model %s (version %s) not found in DB, creating",
                self._model,
                __version__,
            )
        else:
            LOGGER.info("found model %s (version %s) in DB", self._model, __version__)
        return config

    def _retrieve_database_annotations(self, config: Config) -> pd.DataFrame:
        """
        Internal method to retrieve annotations for a single dataset and model.
        :return: pd.DataFrame
        """
        # Get all Query objects for the current dataset
        dataset_queries = (
            Query.select()
            .where(Query.dataset_name_id.in_(Dataset.select().where(Dataset.name == self._dataset)))
            .alias("dataset_queries")
        )

        if self._intent_aware:
            # Get all Triple objects that have ForeignKey relationships to the dataset Query objects, that have Intents
            triples = (
                Triple.select(
                    Triple,
                    Intent.source.alias("intent_source")
                )
                .where(Triple.intent.is_null(False))
                .join(dataset_queries, on=(Triple.query == dataset_queries.c.id))
                .join_from(Triple, Query)
                .join(Intent, on=(Triple.intent == Intent.id))
                .where(Intent.source == self._intent_source)
            )

        else:
            # Get all Triple objects that have ForeignKey relationships to the dataset Query objects, that do not have Intents
            triples = (
                Triple.select()
                .where(Triple.intent.is_null())
                .join(dataset_queries, on=(Triple.query == dataset_queries.c.id))
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
        print(f"Size before dedupe: {annotations.shape[0]}")
        # For some reason, there are duplicates of triples (which by definition should be unique), so manual de-duplication is needed
        annotations = annotations.drop_duplicates(subset=["triple"])
        print(f"Size after dedupe: {annotations.shape[0]}")

        LOGGER.info(
            f"\tLoaded {annotations.shape[0]} judgments.")

        return annotations

    def evaluate(self):
        # Retrieve the Annotation entries from the database (predictions)
        model_annotations = self._retrieve_database_annotations(self._get_config())
        model_annotations = model_annotations[["query_id", "intent_id", "doc_id", "result"]]

        if self._qrels_true_path is not None:
            # Load the human judgments (ground truth) for comparison
            delim = " " if "dl-mia" in self._dataset else "\t"
            human_df = pd.read_csv(Path(self._qrels_true_path), sep=delim,
                                   names=["query_id", "intent_id", "doc_id", "rel"])

            # Filter out any rows that have negative relevance scores
            human_df = human_df[human_df["rel"] >= 0].copy()
            LOGGER.info(f"\tLoaded {human_df.shape[0]} human judgments.")

        else:
            # If no path to a qrels file was given, check the database for the ground truth
            human_annotations = self._retrieve_database_annotations(self._get_config(human=True))
            human_annotations.rename(columns={"result": "rel"}, inplace=True)
            human_df = human_annotations[["query_id", "intent_id", "doc_id", "rel"]]

        assert model_annotations.shape[0] == human_df.shape[0]

        # Sanity check that the judgement from q-d pair to q,a,d triple did not change
        # if "clueweb" in self._dataset:
        #     # Load subsamples
        #     register_subsamples()
        # dataset = ir_datasets.load(str(self._dataset))
        # original_qrels = dataset.qrels_iter()

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