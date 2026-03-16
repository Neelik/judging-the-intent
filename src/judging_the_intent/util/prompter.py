import re
import logging
LOGGER = logging.getLogger(__name__)


def _parser_binary(text):
    if text in ["Relevant", "Irrelevant"]:
        return "1" if text == "Relevant" else "0"

    LOGGER.debug(f"PARSING:\t\t{text}")
    if "Relevant" in text:
        text_ = "1"
    elif "Irrelevant" in text or "Ir" in text:
        text_ = "0"
    else:
        text_ = "0"

    LOGGER.debug(f"PARSED:\t\t{text_}")
    return text_

def _extract_first_digit(text):
    for char in text:
        if char.isdigit():
            return char
    return "0"

def _parser_digit(text):
    if text in ["0", "1", "2", "3", "4"]:
        return text
    else:
        LOGGER.debug(f"PARSING:\t\t{text}")
        text_ = _extract_first_digit(text)
        LOGGER.debug(f"PARSED:\t\t{text_}")
    return text_

def _parse_intents(text: str) -> list:
    LOGGER.debug(f"PARSING:\t\t{text}")
    # There's a variation where the intents come first, then the documents
    # And there's a variation where each item is an intent,document list pair in a numbered list
    split_on_intentions = text.split("Intentions::")[-1]
    split_on_line_break = split_on_intentions.split("\n")[1:]
    intents_list = [item.split("Document_List::")[0].strip().replace(",", "").replace(". ", "")
                    for item in split_on_line_break]
    cleaned_intents = [re.sub(r'\d+', '', text) for text in intents_list]

    return cleaned_intents


class Prompter:
    def __init__(self, prompt_style: str):
        self._allowed_styles = ["binary", "dna", "binary-intent", "dna-intent"]
        if prompt_style in self._allowed_styles:
            self.prompt_style = prompt_style
        else:
            raise NotImplemented(f"Prompt style {prompt_style} is not supported. Allowed styles are "
                                 f"{[s for s in self._allowed_styles]}")

        if self.prompt_style == "binary":
            self.template = """
            Instruction: Please assess the relevance of the provided passage to the following question. Please output "Relevant" or "Irrelevant".
            Question: {question}
            Passage: {passage}
            Output:"""
            self.splitter = "Output:"
            self.pos_label = "Relevant"
            self.neg_label = "Irrelevant"
            self.demonstration = "Question: {question}\nPassage: {passage}\nOutput: {output}\n"
            self.parser = _parser_binary

        elif self.prompt_style == "dna":
            self.template = """You are a search quality rater evaluating the relevance of web pages.
            Given the user query, and a web page, you must provide a score on an integer scale of 0 to 4 to indicate to what extent the given document meets the information needs of the user.
            The scores have the following meanings:
            
            0: fails to meet
            1: slightly meets
            2: moderately meets
            3: highly meets
            4: fully meets
            
            Query: {question}
            Passage: {passage}
            Score:"""
            self.splitter = "Score:"
            self.labels = ["0", "1", "2", "3", "4"]
            self.parser = _parser_digit

        elif self.prompt_style == "binary-intent":
            self.template = """
                        Instruction: Please assess the relevance of the provided passage to the following question. Please output "Relevant" or "Irrelevant".
                        Question: {question}
                        Intent: {intent}
                        Passage: {passage}
                        Output:"""
            self.splitter = "Output:"
            self.pos_label = "Relevant"
            self.neg_label = "Irrelevant"
            self.parser = _parser_binary

        elif self.prompt_style == "dna-intent":
            self.template = """You are a search quality rater evaluating the relevance of web pages.
                        Given the query of the user, user search intent, and a web page, you must provide a score on an integer scale of 0 to 4 to indicate to what extent the given document meets the information needs of the user.
                        The scores have the following meanings:

                        0: fails to meet
                        1: slightly meets
                        2: moderately meets
                        3: highly meets
                        4: fully meets

                        Query: {question}
                        Intent: {intent}
                        Passage: {passage}
                        Score:"""
            self.splitter = "Score:"
            self.labels = ["0", "1", "2", "3", "4"]
            self.parser = _parser_digit


class IntentGenerationPrompter:
    def __init__(self, prompt_style: str):
        self._allowed_styles = ["generate-intent"]
        if prompt_style in self._allowed_styles:
            self.prompt_style = prompt_style
        else:
            raise NotImplemented(f"Prompt style {prompt_style} is not supported.")

        if self.prompt_style == "generate-intent":
            self.template = """A person wants to determine distinct intentions behind a query. Query: {query}. Give five descriptive (max. 15 words) distinct intentions which are easy to understand. Consider all documents in your response. Response should be in the format: Intentions:: <intention>, Document_List:: <list of documents with the intention>.\n\nDocuments: {documents}"""
            self.parser = _parse_intents
