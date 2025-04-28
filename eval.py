import csv
from email.policy import default

import pandas as pd
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import classification_report


def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("ANNOTATION_FILE", help="Annotated scored docs (.tsv)")
    args = ap.parse_args()

    gt = defaultdict(bool)
    print("reading ground truth...")
    with open(
        Path(__file__).parent / "DL-MIA" / "data" / "qid_iid_qrel.txt",
        encoding="utf-8",
        newline="",
    ) as fp:
        for line in fp:
            q_id, i_id, doc_id, rel = line.split()

            # use binary relevances for now
            gt[(q_id, i_id, doc_id)] = int(rel) > 0

    annotations = defaultdict(bool)
    print("reading annotations...")
    with open(args.ANNOTATION_FILE, encoding="utf-8", newline="") as fp:
        for q_id, i_id, doc_id, rel, _ in csv.reader(fp, delimiter="\t"):
            # use binary relevances for now
            annotations[(q_id, i_id, doc_id)] = int(rel) > 0

    print("producing classification report...")
    y_true, y_pred = [], []
    for triple, label in annotations.items():
        y_pred.append(label)
        # defaultdict returns False for missing triples
        y_true.append(gt[triple])

    print(classification_report(y_true, y_pred))

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
    for query in gt_queries:
        for intent in gt_intents:
            gt = ground_truth[(ground_truth["query_id"] == query) & (ground_truth["intent_id"] == intent)]
            pred = predictions[(predictions["query_id"] == query) & (predictions["intent_id"] == intent)]
            if not gt.empty and not pred.empty:
                try:
                    spearman_ranks[(query, intent)] = spearmanr(a=gt.rel.values, b=pred.rel.values)
                except ValueError:
                    incomplete_count += 1
                    incomplete.append((query, intent))

    for key in spearman_ranks.keys():
        print(f"{key}\t\t{spearman_ranks[key]}")

    print(f"There were {incomplete_count} correlations unable to be calculated")


if __name__ == "__main__":
    main()
