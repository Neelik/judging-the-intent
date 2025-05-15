import csv

import pandas as pd
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import classification_report, multilabel_confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm


def relevance_judgment_accuracy(results: pd.DataFrame):
    accuracy = 0.0
    return accuracy



def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("ANNOTATION_FILE", help="Annotated scored docs (.tsv)")
    ap.add_argument("-d", "--dataset", default="DL-MIA", dest="EVAL_DATASET", help="Dataset name from ir_datasets (e.g., 'msmarco-passage-v2). Defaults to DL-MIA")
    args = ap.parse_args()

    gt = defaultdict(bool)
    gt_ml = defaultdict(int)
    print("reading ground truth...")
    if args.EVAL_DATASET == "DL-MIA":
        with open(
            Path(__file__).parent / "DL-MIA" / "data" / "qid_iid_qrel.txt",
            encoding="utf-8",
            newline="",
        ) as fp:
            for line in fp:
                q_id, i_id, doc_id, rel = line.split()

                # use binary relevances for now -- [0,1] --> 0, [2,3] --> 1
                gt[(q_id, i_id, doc_id)] = int(rel) > 1
                gt_ml[(q_id, i_id, doc_id)] = int(rel)
    else:
        raise NotImplemented("Need to build access point for ir_datasets qrels")

    annotations = defaultdict(bool)
    annotations_ml = defaultdict(int)
    print("reading annotations...")
    with open(args.ANNOTATION_FILE, encoding="utf-8", newline="") as fp:
        for q_id, i_id, doc_id, rel, _ in csv.reader(fp, delimiter="\t"):
            # use binary relevances for now -- [0,1] --> 0, [2,3] --> 1
            annotations[(q_id, i_id, doc_id)] = int(rel) > 1
            annotations_ml[(q_id, i_id, doc_id)] = int(rel)

    print("producing binary classification report...")
    y_true, y_pred = [], []
    for triple, label in annotations.items():
        y_pred.append(label)
        # defaultdict returns False for missing triples
        y_true.append(gt[triple])

    print(classification_report(y_true, y_pred))

    print("producing multiclass classification report...")
    ml_y_true, ml_y_pred = [], []
    for triple, label in annotations_ml.items():
        ml_y_pred.append(label)
        # defaultdict returns False for missing triples
        ml_y_true.append(gt_ml[triple])

    print(classification_report(ml_y_true, ml_y_pred, labels=[0, 1, 2, 3]))

    # print(multilabel_confusion_matrix(ml_y_true, ml_y_pred, labels=[0, 1, 2, 3]))
    #
    # print(precision_recall_fscore_support(ml_y_true, ml_y_pred, labels=[0, 1, 2, 3]))

    print()
    print("calculating spearman rank correlation")
    ground_truth = pd.read_csv(Path.cwd().joinpath("DL-MIA", "data", "qid_iid_qrel.txt"), sep=" ",
                               names=["query_id", "intent_id", "doc_id", "rel"])
    predictions = pd.read_csv(Path.cwd().joinpath(args.ANNOTATION_FILE), sep="\t",
                              names=["query_id", "intent_id", "doc_id", "rel", "model"])
    gt_queries = set(ground_truth.query_id.values)
    gt_intents = set(ground_truth.intent_id.values)

    spearman_ranks = defaultdict(list)
    incomplete_count = 0
    incomplete = []
    for query in tqdm(gt_queries):
        for intent in gt_intents:
            gt = ground_truth[(ground_truth["query_id"] == query) & (ground_truth["intent_id"] == intent)]
            pred = predictions[(predictions["query_id"] == query) & (predictions["intent_id"] == intent)]
            assert gt.shape[0] == pred.shape[0]
            if not gt.empty and not pred.empty:
                try:
                    spearman_ranks[(query, intent)] = spearmanr(a=gt.rel.values, b=pred.rel.values)
                except ValueError:
                    incomplete_count += 1
                    incomplete.append((query, intent))

    for key in spearman_ranks.keys():
        print(f"{key[0]}\t{key[1]}\t\t{spearman_ranks[key]}")

    print(f"There were {incomplete_count} correlations unable to be calculated")


if __name__ == "__main__":
    main()
