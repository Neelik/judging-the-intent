import json
import re


class Parser:
    """Extract the relevance score from an LLM output.

    The base parser accepts all models.
    """

    def matches(self, model: str) -> bool:
        """Whether this parser supports the model.

        :param model: The model identifier.
        :return: True if the parser accepts the model, False otherwise.
        """
        return True

    def __call__(self, response_text: str) -> tuple[int, str]:
        """Parse the model response.

        :param response_text: The model output.
        :return: The relevance score.
        """
        loaded_string = json.loads(response_text)
        result = int(loaded_string["Relevance Score"])
        explanation = loaded_string["Explanation"]
        return result, explanation


class Phi4Parser(Parser):
    """Parser for phi4 models.

    Expects to find a JSON snippet with the score anywhere within the response.
    """

    def matches(self, model: str) -> bool:
        return model.startswith("phi4")

    def __call__(self, response_text: str) -> int:
        return int(re.search(r'{"Relevance Score": (\d)}', response_text).group(1))
