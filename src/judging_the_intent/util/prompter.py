import ast
import re
import json
import logging
from abc import ABCMeta, abstractmethod

LOGGER = logging.getLogger(__name__)


def detect_list(text: str, list_style: str) -> list:
    """
        Function to detect different list patterns within a string.

        :param text: Input string to be examined
        :param list_style: String identifying the list pattern to look for, e.g., square-bracketed or numbered.

        `numbered` style detects if a string contains a numbered list pattern.
        Supports formats like:
        1. Item
        2) Item
        3 - Item

        `square-bracketed` style detects if a string contains a square bracketed list pattern that can be parsed as a
        literal Python list, i.e., has spaces after the commas.

        :return: List of items parsed out following the detected patterns
    """
    # Regex pattern for numbered list items
    numbered_pattern = r'^\s*\d+\s*[\.\-\)]\s+.+'
    sq_bracketed_pattern = r'\[.*?\]'

    if list_style == 'numbered':
        # Split text into lines and check each
        lines = text.strip().splitlines()
        matches = [line for line in lines if re.match(numbered_pattern, line)]

        return matches  # Returns list of matching lines

    elif list_style == 'square-bracketed':
        matches = re.findall(sq_bracketed_pattern, text)

        results = []
        for match in matches:
            parsed = None
            try:
                # Try to safely evaluate as a Python literal list
                parsed_value = ast.literal_eval(match)
                if isinstance(parsed_value, list):
                    parsed = parsed_value
            except (SyntaxError, ValueError):
                # Not a valid Python list, keep parsed as None
                pass
            if parsed is not None:
                results.extend(parsed)

        return results[:5]

    else:
        raise NotImplementedError(f"Unsupported list style: {list_style}.")


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


class AnnotationPrompter:
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

class GenerationPrompter(metaclass=ABCMeta):
    def __init__(self, prompt_style: str):
        self.prompt_style = prompt_style
        self.template = None

    @abstractmethod
    def build_prompt(self, **kwargs) -> str:
        raise NotImplemented

    @abstractmethod
    def parse(self, text: str):
        raise NotImplemented

    @staticmethod
    @abstractmethod
    def _set_template() -> str:
        raise NotImplemented

class IntentGenerationPrompter(GenerationPrompter):
    def __init__(self, prompt_style: str):
        super().__init__(prompt_style)
        self._allowed_style = "generate-intent"
        self.template = self._set_template()
        if prompt_style != self._allowed_style:
            raise NotImplemented(f"Prompt style {prompt_style} is not supported.")

    def parse(self, text: str) -> list:
        LOGGER.debug(f"PARSING:\t\t{text}")

        # Version where each intent is preceded by the 'Intentions::' substring
        if text.count("Intentions::") > 1 and "```" not in text:
            has_intent = [split_text if "Document_List" in split_text else None for split_text in
                          text.split("Intentions::")]
            intents_list = [x.strip() for x in has_intent if x is not None]
            cleaned_intents = [x.split("Document_List::")[0].strip().replace(",", "") for x in intents_list]

        # Version where there is a numbered list following the 'Intentions::' substring
        elif text.count("Intentions::") == 1:
            # And there's a variation where each item is an intent,document list pair in a numbered list
            split_on_intentions = text.split("Intentions::")[-1]
            split_on_line_break = split_on_intentions.split("\n")[1:]
            intents_list = [
                item.split("Document_List::")[0].strip().replace(",", "")
                .replace(". ", "").replace("*", "")
                for item in split_on_line_break]
            cleaned_intents = [re.sub(r'\d+', '', text) for text in intents_list]
            cleaned_intents = [ci for ci in cleaned_intents if ci != ""]
        else:
            # JSON variation
            pattern = r"```(?:\w+)?\n(.*?)```"
            matches = re.findall(pattern, text, flags=re.DOTALL)
            if len(matches) > 0:
                # Making the assumption that the first code block that shows up is the one we want
                matched = matches[0]
                matched_to_json = json.loads(matched)
                cleaned_intents = [i["intention"] for i in matched_to_json["intentions"]]
            else:
                cleaned_intents = []

        intent_candidates = cleaned_intents[:5]
        processed_intents = []
        for candidate in intent_candidates:
            updated_text = candidate.replace("_", " ").replace("<", "").replace(">", "")
            updated_text_no_nums = re.sub(r"\d+", "", updated_text)
            updated_text_no_punc = re.sub(r'[^a-zA-Z0-9\s]', '', updated_text_no_nums)
            updated_text = updated_text_no_punc.strip()

            # Remove clueweb variations
            updated_text = re.sub(r"(?i)clueweben", "", updated_text)
            updated_text = re.sub(r"(?i)clueweb", "", updated_text)

            final_text = updated_text.strip().lower()
            processed_intents.append(final_text)

        # Only return 5 items as we only ask for 5 intents
        return processed_intents

    def build_prompt(self, **kwargs) -> str:
        required_fields = ["query", "documents"]
        assert (all(r in kwargs for r in required_fields))
        assert isinstance(kwargs.get("documents"), list)
        return self.template.format(**kwargs)

    @staticmethod
    def _set_template() -> str:
        return """A person wants to determine distinct intentions behind a query. Query: {query}. Give five descriptive (max. 15 words) distinct intentions which are easy to understand. Consider all documents in your response. Response should be in the format: Intentions:: <intention>, Document_List:: <list of documents with the intention>.\n\nDocuments: {documents}"""


class SubtopicGenerationPrompter(GenerationPrompter):
    def __init__(self, prompt_style: str):
        super().__init__(prompt_style)
        self._allowed_style = "generate-subtopic"
        self.template = self._set_template()
        if prompt_style != self._allowed_style:
            raise NotImplemented(f"Prompt style {prompt_style} is not supported.")

    def parse(self, text: str):
        # Step 1: Split on the output tag - 'Output:'
        output_split = [s for s in text.split("Output:") if s.strip() != '']
        if len(output_split) > 1:
            LOGGER.error(f"Parsing failed for text: {text}: More than one 'Output:' tag in string.")
            return []
        else:
            # Step 2: Handle detecting square-bracketed or numbered lists, looking for former first
            subtopics_string = output_split[0]
            bracketed_subtopics = detect_list(subtopics_string, list_style="square-bracketed")
            if len(bracketed_subtopics) > 0:
                # Found a string representation of a bracketed list, let's parse this as the subtopics
                post_processed = [re.sub(r"[^\w\s-]", "", br.strip().lower(), flags=re.UNICODE)
                                  for br in bracketed_subtopics]
                return post_processed
            else:
                # No bracketed list, then look for a numbered list
                numbered_subtopics = detect_list(subtopics_string, list_style="numbered")
                if len(numbered_subtopics) > 0:
                    # Found a numbered list, treat them as the subtopics
                    no_punc = [re.sub(r"[^\w\s-]", "", nr.strip().lower(), flags=re.UNICODE)
                               for nr in numbered_subtopics]
                    post_processed = [re.sub(r"\d+", "", np) for np in no_punc]
                    return post_processed
                else:
                    # Try last ditch effort that the model simply gave a comma separated list in the 'Output:'
                    comma_separated = subtopics_string.split(",")
                    if len(comma_separated) == 0:
                        LOGGER.error(f"Parsing failed for text: {text}: No subtopics detected.")
                        return []
                    else:
                        # Parse the punctuation and the lowercase
                        post_processed = [re.sub(r"[^\w\s-]", "", nr.strip().lower(), flags=re.UNICODE)
                                          for nr in numbered_subtopics]
                        return post_processed

    def build_prompt(self, **kwargs) -> str:
        required_fields = ["query"]
        # We don't care about the documents, so we drop them beforehand
        kwargs.pop("documents", None)
        assert (all(r in kwargs for r in required_fields))
        return self.template.format(**kwargs)

    @staticmethod
    def _set_template() -> str:
        return """You are an expert at interpreting and refining user queries. 
Given the original query, provide five rewrites that are a more detailed, precise, and contextually enriched version that:
1. Clarifies the intent.
2. Expands relevant keywords and synonyms.
3. Removes ambiguity.
4. Makes it suitable for accurate retrieval or processing.

Original Query: "{query}"

The output should be in the format "Output: [rewrite, rewrite, rewrite]"
Output:"""
        # return """A person wants to determine distinct intentions behind a query. Query: {query}. Give five descriptive (max. 15 words) distinct intentions which are easy to understand and in the form of a question. The response should in the format "Output: [intent, intent, intent]."""

    # """You are an expert at interpreting and refining user queries.
    # Given the original query, provide five rewrites that are a more detailed, precise, and contextually enriched version that:
    # 1. Clarifies the intent.
    # 2. Expands relevant keywords and synonyms.
    # 3. Removes ambiguity.
    # 4. Preserves the original meaning.
    # 5. Makes it suitable for accurate retrieval or processing.
    #
    # Original Query: "{query}"
    #
    # The output should be in the format "Output: [rewrite, rewrite, rewrite]"
    # Output:"""