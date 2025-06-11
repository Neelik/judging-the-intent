# judging-the-intent

## Setup

The easiest way to run the scripts is by using [`uv`](https://docs.astral.sh/uv/). After `uv` has been installed, clone the repository and create a virtual environment:

```bash
uv sync
```

## Usage

First, run `create_db` in order to create an SQLite database containing queries, intents, and documents. For example:

```bash
uv run python -m judging_the_intent.create_db --datasets corpus-subsamples/clueweb09/en/trec-web-2009 corpus-subsamples/clueweb09/en/trec-web-2010 --data_dir trec-web/ --db_file data.db
```

Afterwards, make sure that Ollama is running and the models are available and run `annotate`:

```bash
uv run python -m judging_the_intent.annotate --models mistral:7b-instruct --db_file data.db
```
