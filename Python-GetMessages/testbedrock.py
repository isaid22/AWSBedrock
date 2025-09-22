import json
import logging
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import yaml
import time
import pandas as pd 

def extract_messages_from_converse(response: dict) -> list[str]:
    """Extract a list of messages from a Bedrock Converse API response.

    Handles both JSON-array outputs and plain text with newlines/bullets.
    Returns a list of strings (messages). If nothing is found, returns [].
    """
    if not isinstance(response, dict):
        return []

    # Nova models return under response["output"]["message"]["content"][i]["text"]
    texts: list[str] = []
    try:
        output = response.get("output", {})
        message = output.get("message", {})
        content = message.get("content", [])
        for part in content:
            if isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
                texts.append(part["text"].strip())
    except Exception:
        pass

    # Fallback for models that may use 'results' or 'completion' shapes (defensive only)
    if not texts:
        if isinstance(response.get("results"), list):
            for r in response["results"]:
                t = r.get("outputText") or r.get("text")
                if isinstance(t, str):
                    texts.append(t.strip())
        elif isinstance(response.get("completion"), str):
            texts.append(response["completion"].strip())

    if not texts:
        return []

    # Often we asked the model to return a JSON array of strings. Try parsing that first.
    import json as _json
    messages: list[str] = []
    for t in texts:
        t_stripped = t.strip()
        parsed = None
        if t_stripped.startswith("[") and t_stripped.endswith("]"):
            try:
                parsed = _json.loads(t_stripped)
            except Exception:
                parsed = None
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            messages.extend([x.strip() for x in parsed if x and x.strip()])
        else:
            # Split plain text into potential lines; remove bullets and empties
            for line in t_stripped.splitlines():
                line = line.strip()
                if not line:
                    continue
                # remove common list prefixes like '1.', '-', '*'
                if line[:2].isdigit() and line[2:3] == ".":
                    line = line[3:].strip()
                if line.startswith(("- ", "* ", "• ")):
                    line = line[2:].strip()
                messages.append(line)

    # Deduplicate while preserving order
    seen = set()
    unique_messages = []
    for m in messages:
        if m not in seen:
            seen.add(m)
            unique_messages.append(m)

    # If there are more than 10, take the first 10; if fewer, return what we have
    return unique_messages[:10]

def extract_messages_from_converse_response(resp):
    """
    Extract a list of messages (strings) from a Bedrock Converse API response.

    Supports both formats:
    - resp['output']['text'] -> a JSON array string or raw text containing messages
    - resp['output']['message']['content'][i]['text'] -> text blocks that may contain JSON array

    Returns: list[str] (up to 10 items)
    """
    import re

    output = (resp or {}).get('output') or {}

    raw_text_blocks = []
    # Case 1: output.text present
    if isinstance(output.get('text'), str):
        raw_text_blocks.append(output['text'])

    # Case 2: output.message.content[].text present
    message = output.get('message') or {}
    content = message.get('content') or []
    for item in content:
        t = item.get('text')
        if isinstance(t, str):
            raw_text_blocks.append(t)

    raw = "\n".join(raw_text_blocks).strip()
    if not raw:
        return []

    # Try to parse as a JSON array of strings directly
    try:
        data = json.loads(raw)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data[:10]
        if isinstance(data, dict):
            for key in ('messages', 'output', 'data', 'result'):
                v = data.get(key)
                if isinstance(v, list) and all(isinstance(x, str) for x in v):
                    return v[:10]
    except Exception:
        pass

    # Try to locate a JSON array substring in the text and parse it
    m = re.search(r'\[\s*"(?:\\.|[^"\\])*"(?:\s*,\s*"(?:\\.|[^"\\])*")*\s*\]', raw, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return arr[:10]
        except Exception:
            pass

    # Fallback: split lines, strip bullets/numbers, and dedupe
    candidates = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^\s*[\-\*\u2022]\s*', '', line)  # bullets
        line = re.sub(r'^\s*\d+[\.)]\s*', '', line)       # numbering
        if len(line) >= 2 and ((line[0] == line[-1] == '"') or (line[0] == line[-1] == "'")):
            line = line[1:-1]
        if line:
            candidates.append(line)

    seen = set()
    deduped = []
    for c in candidates:
        s = c.strip()
        if s and s not in seen:
            seen.add(s)
            deduped.append(s)

    return deduped[:10]

def main(): 


    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # Create a Bedrock client
    bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    model_id = 'amazon.nova-micro-v1:0'
    system_prompt = "You are an expert in promoting banking customer with either home mortgage purchase, refinancing or Home Equity Line of Credit. I need you to come up with 10 brief and powerful messages, each message have at most twenty words. Use courteous and professional tone. Do not make any offers or mention anything numeric, such as years, terms, interest rates, fees."
    user_message = "This message is for user that is interested in home equity loan."
    json_output = f"""{
        user_message} 
        
        Return the output as JSON array of strings, with each string containing one message. Do not include any extra text.

    """
    responses = []
    latencies = []


    inference_configuration = {
        "maxTokens": 1024,
        "temperature": 0.9,
        "topP": 0.9
    }

    response = bedrock_client.converse(
        modelId = model_id,
        system = [
            {"text": system_prompt}
        ],
        messages = [
            {
                "role": "user",
                "content": [{"text": json_output}]
            }
        ],
        inferenceConfig = inference_configuration
    )

    # Extract and print messages cleanly
    messages = extract_messages_from_converse(response)
    if not messages:
        print("No messages parsed. Raw response follows:\n")
        print(json.dumps(response, indent=2, default=str))
    else:
        print("Messages ({}):".format(len(messages)))
        for i, m in enumerate(messages, 1):
            print(f"{i}. {m}")

    # Extract the ten short messages from the response and print them
    messages = extract_messages_from_converse_response(response)
    if messages:
        print("\nExtracted messages:")
        for i, msg in enumerate(messages, start=1):
            print(f"{i}. {msg}")
    else:
        print("\nNo messages could be extracted from the response.")




if __name__ == "__main__":
    main()
    