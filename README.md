# judging-the-intent

## Setup

The easiest way to run the scripts is by using [`uv`](https://docs.astral.sh/uv/). After `uv` has been installed, clone the repository and create a virtual environment:

```bash
uv sync
```

## Usage

The scripts expect the following environment variables to be set:

- `POSTGRES_DB`
- `POSTGRES_USER`
- `POSTGRES_PASSWORD`
- `POSTGRES_HOST`
- `POSTGRES_PORT`
- `OLLAMA_HOST`

First, run `create_db` in order to initialize the database with queries, intents, and documents. For example:

```bash
uv run python -m judging_the_intent.create_db --datasets corpus-subsamples/clueweb09/en/trec-web-2009 corpus-subsamples/clueweb09/en/trec-web-2010 --data_dir trec-web/
```

Afterwards, make sure that Ollama is running and the models are available and run `annotate`:

```bash
uv run python -m judging_the_intent.annotate --models llama3.1:8b-instruct-q4_K_M mistral:7b-instruct-v0.3-q4_0
```
