import csv
import pandas as pd
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import classification_report
from tqdm import tqdm


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
            for line in tqdm(fp):
                q_id, i_id, doc_id, rel = line.split()

                # use binary relevances for now -- [0,1] --> 0, [2,3] --> 1
                gt[(q_id, i_id, doc_id)] = round(float(rel)) > 1
                gt_ml[(q_id, i_id, doc_id)] = round(float(rel))
    else:
        with open(
            Path(__file__).parent.joinpath("trec-web", "qrels", f"{args.EVAL_DATASET.replace('/', '=')}-filtered-qrels.tsv"),
            encoding="utf-8",
            newline=""
        ) as fp:
            for q_id, i_id, doc_id, rel in tqdm(csv.reader(fp, delimiter="\t")):

                # use binary relevances for now -- [0,1] --> 0, [2,3] --> 1
                gt[(q_id, i_id, doc_id)] = round(float(rel)) > 1
                gt_ml[(q_id, i_id, doc_id)] = round(float(rel))

    annotations = defaultdict(bool)
    annotations_ml = defaultdict(int)
    print("reading annotations...")
    with open(args.ANNOTATION_FILE, encoding="utf-8", newline="") as fp:
        for q_id, i_id, doc_id, rel, _ in tqdm(csv.reader(fp, delimiter="\t")):
            # use binary relevances for now -- [0,1] --> 0, [2,3] --> 1
            annotations[(q_id, i_id, doc_id)] = round(float(rel)) > 1
            annotations_ml[(q_id, i_id, doc_id)] = round(float(rel))

    print("producing binary classification report...")
    y_true, y_pred = [], []
    for triple, label in tqdm(annotations.items()):
        y_pred.append(label)
        # defaultdict returns False for missing triples
        y_true.append(gt[triple])

    print(classification_report(y_true, y_pred))

    print("producing multiclass classification report...")
    ml_y_true, ml_y_pred = [], []
    for triple, label in tqdm(annotations_ml.items()):
        ml_y_pred.append(label)
        # defaultdict returns False for missing triples
        ml_y_true.append(gt_ml[triple])

    print(classification_report(ml_y_true, ml_y_pred, labels=[0, 1, 2, 3]))


if __name__ == "__main__":
    main()
