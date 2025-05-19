import json
from collections.abc import Generator, Iterable, Mapping
from typing import Any
from parsers import phi4
import os

import requests

OLLAMA_PORT = os.environ.get("OLLAMA_HOST")
OLLAMA_API = f"http://localhost:{OLLAMA_PORT}/api"


class OllamaTripleAnnotator:
    """
    Wrapper class allowing for inference calls to Ollama models using the DNA prompt format
    :param model: Name of the Ollama model to be used in inference
    :param include_intent: Flag that identifies whether to include search intent in the prompt
    :param triples: Iterable triples ((query_id, query), (intent_id, intent), (document_id, document))
    """

    def __init__(
        self,
        model: str,
        include_intent: bool,
        triples: Generator[tuple[str, str], tuple[str, str], tuple[str, str]],
    ) -> None:
        self.triples = triples
        self.model = model
        self.include_intent = include_intent

    def configure(self) -> None:
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

    def get_judgments(self, stream=False) -> Iterable[Mapping[str, Any]]:
        """
        Method to perform relevance judgment inference over the Ollama local API

        :param stream: Determines streaming behavior for Ollama api call - See https://github.com/ollama/ollama/blob/main/docs/api.md for details
        :yield: IDs, predicted relevance score, status code, model response
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

        for (q_id, q), (i_id, i), (d_id, d) in self.triples:
            if self.include_intent:
                self._build_prompt(q, i, d)
            else:
                self._build_prompt(q, "", d)
            data = {"prompt": self.prompt, "model": self.model, "stream": stream}
            json_data = json.dumps(data)
            response = requests.post(
                url=f"{OLLAMA_API}/generate",
                data=json_data,
                headers=ollama_local_headers,
            )

            result = {
                "query_id": q_id,
                "intent_id": i_id,
                "doc_id": d_id,
                "status_code": response.status_code,
                "model_response": json.loads(response.text),
            }

            # Handle the response
            try:
                if self.model == "phi4":
                    json_response = phi4.parse_response(result["model_response"]["response"])
                    result["relevance_score"] = json_response["Relevance Score"]
                else:
                    result["relevance_score"] = json.loads(
                        result["model_response"]["response"]
                    )["Relevance Score"]
            except Exception as ex:
                result["error"] = repr(ex)
            finally:
                yield result

    def _build_prompt(self, query: str, intent: str, doc: str) -> None:
        """
        Internal class method to inject query, intent, and document values into the DNA prompt to be used in model inference

        :param query: Query to be injected into DNA prompt
        :param intent: Intent to be injected into DNA prompt
        :param doc: Document to be injected into DNA prompt
        :return: None
        """
        if intent == "":
            self.prompt = f'You are a search quality rater evaluating the relevance of web pages.\nGiven a query and a text, you must provide a score on an integer scale of 0 to 3 with the following meanings:\n3=Perfectly relevant: The passage is dedicated to the query and contains the exact answer.\n2=Highly relevant: The passage matches the query and has some answer, but the answer may be a bit unclear, or hidden amongst extraneous information.\n1=Related: The passage seems related to the query but does not answer it.\n0=Irrelevant: The passage has nothing to do with the query.\nAssume that you are writing a report on the subject of the query.\n\nQuery: A person has typed "{query}" into a search engine.\nConsider the following web page.\n-BEGIN WEB PAGE CONTENT-\n{doc}\n-END WEB PAGE CONTENT-\n\nInstructions:\nProduce a single relevance score in json format without providing any reasoning. (Example: {{"Relevance Score": 1}})\n\nYour answer:\n'
        else:
            self.prompt = f'You are a search quality rater evaluating the relevance of web pages.\nGiven a query, an intent description, and a text, you must provide a score on an integer scale of 0 to 3 with the following meanings:\n3=Perfectly relevant: The passage is dedicated to the intent and contains the exact answer for the query.\n2=Highly relevant: The passage matches the intent and has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.\n1=Related: The passage seems related to the query and intent but does not answer it.\n0=Irrelevant: The passage has nothing to do with the query or the intent.\nAssume that you are writing a report on the subject of the query.\n\nQuery: A person has typed "{query}" into a search engine. They were looking for "{intent}"\nConsider the following web page.\n-BEGIN WEB PAGE CONTENT-\n{doc}\n-END WEB PAGE CONTENT-\n\nInstructions:\nProduce a single relevance score in json format without providing any reasoning. (Example: {{"Relevance Score": 1}})\n\nYour answer:\n'

    def _is_model_available(self) -> bool:
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

    def unload(self):
        """
        Method to set keep-alive for the Ollama model to 0, removing it from memory. Useful when the class is being run
        on a shared resource.

        :return: boolean of if the request gave a 200
        """
        ollama_local_headers = {"Content-Type": "application/json"}
        data = {"model": self.model, "keep_alive": 0}
        json_data = json.dumps(data)
        response = requests.post(
            url=f"{OLLAMA_API}/generate",
            data=json_data,
            headers=ollama_local_headers,
        )

        return response.status_code == 200