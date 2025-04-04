from dna_prompt import OllamaTripleAnnotator
import ir_datasets

def single_instance_example(dataset):
    documents = dataset.docs_iter()
    queries = dataset.queries_iter()
    print("Testing iterable of length 1, i.e., single item inference")
    for q in queries[:1]:
        for d in documents[:1]:
            query_text = [q.text]
            passage_text = [d.text]
            intent = ["learn about digestion and metabolism process"]
            single_annotator = OllamaTripleAnnotator("mistral", query_text, intent, passage_text)
            single_annotator.configure()
            single_annotator.get_judgments()
            single_annotator.display_judgements()


def multi_instance_example(dataset):
    documents = dataset.docs_iter()
    queries = dataset.queries_iter()
    intents = ["learn about digestion and metabolism process"] * 5
    print("Testing iterable of length greater than one")
    annotator = OllamaTripleAnnotator("mistral", list(queries[:5]), intents, list(documents[:5]))
    annotator.configure()
    annotator.get_judgments()
    annotator.display_judgements()


def model_not_available_example(dataset):
    documents = dataset.docs_iter()
    queries = dataset.queries_iter()
    intents = ["learn about digestion and metabolism process"] * 5
    annotator = OllamaTripleAnnotator("gibberish", list(queries[:5]), intents, list(documents[:5]))
    annotator.configure()
    annotator.get_judgments()
    annotator.display_judgements()


def iterables_different_size_example(dataset):
    documents = dataset.docs_iter()
    queries = dataset.queries_iter()
    intents = ["learn about digestion and metabolism process"] * 5
    print("Testing iterable of length greater than one")
    annotator = OllamaTripleAnnotator("mistral", list(queries[:10]), intents, list(documents[:5]))
    annotator.configure()
    annotator.get_judgments()
    annotator.display_judgements()


if __name__ == "__main__":
    dataset = ir_datasets.load("msmarco-passage-v2/trec-dl-2023")
    try:
        model_not_available_example(dataset)
    except AssertionError as e:
        print(e.args)
    print("\n", "=" * 30)
    try:
        iterables_different_size_example(dataset)
    except AssertionError as e:
        print(e.args)
    print("\n", "=" * 30)
    single_instance_example(dataset)
    print("\n", "=" * 30)
    multi_instance_example(dataset)
