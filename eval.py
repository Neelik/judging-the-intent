import csv
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import defaultdict
from pathlib import Path

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

    annotations = defaultdict(int)
    print("reading annotations...")
    with open(args.ANNOTATION_FILE, encoding="utf-8", newline="") as fp:
        for q_id, i_id, doc_id, rel, _ in csv.reader(fp, delimiter="\t"):
            # use binary relevances for now
            annotations[(q_id, i_id, doc_id)] = int(rel) > 0

    y_true, y_pred = [], []
    for triple, label in annotations.items():
        y_pred.append(label)
        # defaultdict returns False for missing triples
        y_true.append(gt[triple])

    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
