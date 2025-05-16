import csv
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from ir_datasets_subsample import register_subsamples
from pathlib import Path
import ir_datasets
from tqdm import tqdm
from dna_prompt import OllamaTripleAnnotator


def main():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("MODEL", help="Ollama model")
    ap.add_argument("--d", dest="DATASET", help="Name of ir_dataset to perform annotation on.")
    ap.add_argument("--out_file", default="out.tsv", help="Output file (.tsv)")
    ap.add_argument("--i", action='store_true', help="Include the intent in the prompt to the LLM.")
    args = ap.parse_args()

    print("reading intents...")
    print("creating intent lookup...")
    if args.DATASET.lower() == "dl-mia":
        with open(
                Path(__file__).parent / "DL-MIA" / "data" / "intent.tsv",
                encoding="utf-8",
                newline="",
        ) as fp:
            intents = {row[0]: row[1] for row in csv.reader(fp, delimiter="\t")}

        print("creating query lookup...")
        with open(
                Path(__file__).parent / "DL-MIA" / "data" / "query.tsv",
                encoding="utf-8",
                newline="",
        ) as fp:
            queries = {row[0]: row[1] for row in csv.reader(fp, delimiter="\t")}
    else:
        # Need to load the dataset from ir_datasets, or the subsample, and collect intents and queries
        with open(
                Path(__file__).parent.joinpath("trec-web", f"{args.DATASET.replace('/', '-')}-queries.tsv"),
                encoding="utf-8",
                newline=""
        ) as fp:
            queries = {row[0]: row[1] for row in csv.reader(fp, delimiter="\t")}

        with open(
                Path(__file__).parent.joinpath("trec-web", f"{args.DATASET.replace('/', '-')}-qid-iid-intent.tsv"),
                encoding="utf-8",
                newline="",
        ) as fp:
            intents = {row[1]: row[2] for row in csv.reader(fp, delimiter="\t")}

    def triple_generator(q, i):
        if args.DATASET.lower() == "dl-mia":
            with open(
                    Path(__file__).parent / "DL-MIA" / "data" / "qid_iid_qrel.txt",
                    encoding="utf-8",
                    newline="",
            ) as infile:
                for row in csv.reader(infile, delimiter=" "):
                    dataset = ir_datasets.load("msmarco-passage-v2")
                    docs_store = dataset.docs_store()

                    yield (
                        (row[0], q[row[0]]),
                        (row[1], i[row[1]]),
                        (row[2], docs_store.get(row[2]).text),
                    )

        else:
            dataset = ir_datasets.load(args.DATASET)
            docs_store = dataset.docs_store()
            with open(
                    Path(__file__).parent.joinpath("trec-web", "qrels", f"{args.DATASET.replace('/', '-')}-filtered-qrels.tsv"),
                    encoding="utf-8",
                    newline="",
            ) as infile:
                for row in csv.reader(infile, delimiter="\t"):
                    try:
                        doc_text = docs_store.get(row[2]).text
                    except KeyError:
                        with open (Path(__file__).parent.joinpath(f"{args.DATASET.replace('/', '-')}-{args.MODEL}-docstore-errors.txt"), "a") as errorfile:
                            errorfile.write(f"{row[0]}\t{row[1]}\t{row[2]}\t-3\tDocument Not in DocStore\n")
                        continue
                    yield (
                        (row[0], q[row[0]]),
                        (row[1], i[row[1]]),
                        (row[2], doc_text),
                    )

    annotator = OllamaTripleAnnotator(args.MODEL, args.i, triple_generator(queries, intents))
    annotator.configure()

    with open(args.out_file, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        for j in tqdm(annotator.get_judgments(), desc="Annotating >> "):
            if "error" in j:
                print(
                    f"error for query {j['query_id']}, intent {j['intent_id']}, document {j['doc_id']}: {j['error']}"
                )
            else:
                writer.writerow(
                    [
                        j["query_id"],
                        j["intent_id"],
                        j["doc_id"],
                        j["relevance_score"],
                        args.MODEL,
                    ]
                )

    print(annotator.unload())


if __name__ == "__main__":
    register_subsamples()
    main()