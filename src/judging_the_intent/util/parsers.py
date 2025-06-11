def parse_phi4(response):
    # Expect to find "```json\n{\"Relevance Score\": 2}\n```" as the passed string, parse it down to the relevance score
    # and rebuild the expected object
    first_split = response.split("{")
    second_split = first_split[1].split("}")
    third_split = second_split[0].split(":")
    proper_json = {"Relevance Score": int(third_split[1].strip())}
    return proper_json
