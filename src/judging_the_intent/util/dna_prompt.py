DNA_PROMPT_NO_INTENT = """You are a search quality rater evaluating the relevance of web pages.

Given a query and a text, you must provide a score on an integer scale of 0 to 3 with the following meanings:
3=Perfectly relevant: The passage is dedicated to the query and contains the exact answer.
2=Highly relevant: The passage matches the query and has some answer, but the answer may be a bit unclear, or hidden amongst extraneous information.
1=Related: The passage seems related to the query but does not answer it.
0=Irrelevant: The passage has nothing to do with the query.

Assume that you are writing a report on the subject of the query.

Query: A person has typed "{query}" into a search engine.

Consider the following web page.
-BEGIN WEB PAGE CONTENT-
{doc}
-END WEB PAGE CONTENT-

Instructions:
Produce a single relevance score in json format without providing any reasoning (example: {{"Relevance Score": 1}}).

Your answer:"""

DNA_PROMPT_NO_INTENT_WITH_EXPLANATION = """You are a search quality rater evaluating the relevance of web pages.

Given a query and a text, you must provide a score on an integer scale of 0 to 3 with the following meanings:
3=Perfectly relevant: The passage is dedicated to the query and contains the exact answer.
2=Highly relevant: The passage matches the query and has some answer, but the answer may be a bit unclear, or hidden amongst extraneous information.
1=Related: The passage seems related to the query but does not answer it.
0=Irrelevant: The passage has nothing to do with the query.

Assume that you are writing a report on the subject of the query.

Query: A person has typed "{query}" into a search engine.

Consider the following web page.
-BEGIN WEB PAGE CONTENT-
{doc}
-END WEB PAGE CONTENT-

Instructions:
Produce a single relevance score in json format with reasoning (example: {{"Relevance Score": 1, "Explanation": "Related."}}).

Your answer:"""

DNA_PROMPT_WITH_INTENT = """You are a search quality rater evaluating the relevance of web pages.

Given a query, an intent description, and a text, you must provide a score on an integer scale of 0 to 3 with the following meanings:
3=Perfectly relevant: The passage is dedicated to the intent and contains the exact answer for the query.
2=Highly relevant: The passage matches the intent and has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.
1=Related: The passage seems related to the query and intent but does not answer it.
0=Irrelevant: The passage has nothing to do with the query or the intent.

Assume that you are writing a report on the subject of the query.

Query: A person has typed "{query}" into a search engine. They were looking for "{intent}".

Consider the following web page.
-BEGIN WEB PAGE CONTENT-
{doc}
-END WEB PAGE CONTENT-

Instructions:
Produce a single relevance score in json format without providing any reasoning (example: {{"Relevance Score": 1}}).

Your answer:"""

DNA_PROMPT_WITH_INTENT_WITH_EXPLANATION = """You are a search quality rater evaluating the relevance of web pages.

Given a query, an intent description, and a text, you must provide a score on an integer scale of 0 to 3 with the following meanings:
3=Perfectly relevant: The passage is dedicated to the intent and contains the exact answer for the query.
2=Highly relevant: The passage matches the intent and has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.
1=Related: The passage seems related to the query and intent but does not answer it.
0=Irrelevant: The passage has nothing to do with the query or the intent.

Assume that you are writing a report on the subject of the query.

Query: A person has typed "{query}" into a search engine. They were looking for "{intent}".

Consider the following web page.
-BEGIN WEB PAGE CONTENT-
{doc}
-END WEB PAGE CONTENT-

Instructions:
Produce a single relevance score in json format with reasoning (example: {{"Relevance Score": 1, "Explanation": "Related."}}).

Your answer:"""


def build_prompt(query: str, intent: str | None, doc: str, version="default") -> str:
    """
    Inject query, intent, and document values into the DNA prompt to be used in model inference.

    :param query: Query to be injected into DNA prompt.
    :param intent: Intent to be injected into DNA prompt (if any).
    :param doc: Document to be injected into DNA prompt.
    :param version: Determines whether to request reasoning or not. Options are "default" for no reasoning, "verbose" for including reasoning.
    :return: The assembled LLM prompt.
    """
    if intent is None:
        if version == "verbose":
            return DNA_PROMPT_NO_INTENT_WITH_EXPLANATION.format(query=query, doc=doc)
        else:
            return DNA_PROMPT_NO_INTENT.format(query=query, doc=doc)
    if version == "verbose":
        return DNA_PROMPT_WITH_INTENT_WITH_EXPLANATION.format(query=query, intent=intent, doc=doc)
    else:
        return DNA_PROMPT_WITH_INTENT.format(query=query, intent=intent, doc=doc)
