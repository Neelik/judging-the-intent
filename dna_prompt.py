import json
from collections.abc import Iterable

import requests

OLLAMA_API = "http://localhost:11434/api"


class OllamaTripleAnnotator:
    """
    Wrapper class allowing for inference calls to Ollama models using the DNA prompt format
    :param model: Name of the Ollama model to be used in inference
    :param triples: Iterable triples (query, intent, document)

    :attribute judgements: Python list of dictionaries containing the model-provided relevance judgements
    """

    def __init__(self, model: str, triples: Iterable[str, str, str]) -> None:
        self.triples = triples
        self.judgements = []
        self.model = model

    def configure(self):
        """
        Method to check if the desired Ollama model is available locally. Installs the model if it is not found.

        :return: None
        """
        status = False
        ollama_local_headers = {"Content-Type": "application/json"}
        # First, check if the model is already available locally
        try:
            assert self._is_model_available()
            status = True
        except AssertionError:
            # If not available locally, pull it from upstream
            response = requests.post(
                f"{OLLAMA_API}/pull",
                headers=ollama_local_headers,
                data=json.dumps({"model": self.model, "stream": False}),
            )
            if response.status_code == 200:
                status = True

        # Inform user configuration is completed
        if status:
            print(
                f'Configuration Successful! Model "{self.model}" now available for use.'
            )
        else:
            print(
                f"Configuration Failed! Is the model name ({self.model}) available in the Ollama library? You can check at https://ollama.com/search"
            )

    def get_judgments(self, stream=False) -> Iterable[str]:
        """
        Method to perform relevance judgment inference over the Ollama local API

        :param stream: Determines streaming behavior for Ollama api call - See https://github.com/ollama/ollama/blob/main/docs/api.md for details
        :yield: The model responses
        """
        ollama_local_headers = {"Content-Type": "application/json"}
        # Check if the default ollama model is available locally, if not prompt user to call configure()
        try:
            assert self._is_model_available()
        except AssertionError as e:
            errmsg = f"Model {self.model} not installed -- did you call configure()?"
            args = e.args
            if not args:
                arg0 = errmsg
            else:
                arg0 = f"{args[0]}\n{errmsg}"
            e.args = (arg0,) + args[1:]
            raise

        for trip in self.triples:
            self._build_prompt(*trip)
            data = {"prompt": self.prompt, "model": self.model, "stream": stream}
            json_data = json.dumps(data)
            response = requests.post(
                url=f"{OLLAMA_API}/generate",
                data=json_data,
                headers=ollama_local_headers,
            )

            # Handle the response
            if response.status_code == 200:
                # Successful call, process LLM response
                response_json = json.loads(response.text)
                model_message = response_json["response"]
                yield model_message
            else:
                response_json = json.loads(response.text)
                model_message = response_json["response"]
                yield {"error": model_message}

    def _build_prompt(self, query, intent, doc):
        """
        Internal class method to inject query, intent, and document values into the DNA prompt to be used in model inference

        :param query: Query to be injected into DNA prompt
        :param intent: Intent to be injected into DNA prompt
        :param doc: Document to be injected into DNA prompt
        :return: None
        """
        self.prompt = f'You are a search quality rater evaluating the relevance of web pages.\nGiven a query, an intent description, and a text, you must provide a score on an integer scale of 0 to 3 with the following meanings:\n3=Perfectly relevant: The passage is dedicated to the intent and contains the exact answer for the query.\n2=Highly relevant: The passage matches the intent and has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.\n1=Related: The passage seems related to the query and intent but does not answer it.\n0=Irrelevant: The passage has nothing to do with the query or the intent.\nAssume that you are writing a report on the subject of the query.\n\nQuery: A person has typed "{query}" into a search engine. They were looking for "{intent}"\nConsider the following web page.\n-BEGIN WEB PAGE CONTENT-\n{doc}\n-END WEB PAGE CONTENT-\n\nInstructions:\nProduce a single relevance score in json format without providing any reasoning. (Example: {{"Relevance Score": 1}})\n\nYour answer:\n'

    def _is_model_available(self):
        """
        Method to check for the availability of the desired Ollama model in the local environment

        :return: True if model found, False otherwise
        """
        available = False
        ollama_local_headers = {"Content-Type": "application/json"}
        response = requests.get(f"{OLLAMA_API}/tags", headers=ollama_local_headers)
        if response.status_code == 200:
            response_json = json.loads(response.text)
            names = [rj["name"] for rj in response_json["models"]]
            model_name_split = self.model.split(":")
            if len(model_name_split) == 1:
                # Check only model, ignore tag
                names_split = [n.split(":")[0] for n in names]
                if self.model in names_split:
                    available = True
            else:
                # Check both model and tag
                if self.model in names:
                    available = True
        return available
