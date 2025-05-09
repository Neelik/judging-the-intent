def parse_response(response):
    # Grab the content and decode it
    decoded_content = response.content.decode(response.encoding)

    # Expect to find "```json\n{\"Relevance Score\": 2}\n```" as the decoded string, parse it down to the relevance score
    # and rebuild the expected object
    first_split = decoded_content.split("{")
    second_split = first_split[1].split("}")
    third_split = second_split[0].split(":")
    proper_json = {"Relevance Score": int(third_split[1].strip())}

    return proper_json

