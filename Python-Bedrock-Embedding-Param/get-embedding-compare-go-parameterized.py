import os
import sys
import json
import time
import boto3
import yaml

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load config.yaml
    config = load_config('config.yaml')
    region = config.get('region', 'us-east-1')
    model_id = config.get('model_id', 'amazon.titan-embed-text-v1')
    embedding_length = config.get('embedding_length')
    input_text = config.get('input_text')

    # Allow override from command line
    if len(sys.argv) > 1:
        input_text = sys.argv[1]

    payload = {
        "inputText": input_text
    }

    # Create Bedrock Runtime client
    brc = boto3.client("bedrock-runtime", region_name=region)

    # Convert payload to JSON
    payload_bytes = json.dumps(payload).encode("utf-8")

    # Call Bedrock model and time it
    start = time.time()
    response = brc.invoke_model(
        body=payload_bytes,
        modelId=model_id,
        contentType="application/json"
    )
    elapsed = (time.time() - start) * 1000  # milliseconds

    # Unmarshal response and time it
    start_unmarshal = time.time()
    resp = json.loads(response["body"].read())
    unmarshal_elapsed = (time.time() - start_unmarshal) * 1_000_000  # microseconds

    print("embedding vector from LLM\n", resp["embedding"])
    print()
    print("generated embedding for input -", input_text)
    print("generated vector length -", len(resp["embedding"]))
    print(f"Bedrock model call took: {elapsed:.2f} ms")
    print(f"Unmarshal took: {unmarshal_elapsed:.0f} µs")

    if embedding_length and len(resp["embedding"]) != embedding_length:
        print(f"Warning: embedding length ({len(resp['embedding'])}) does not match expected ({embedding_length})")

if __name__ == "__main__":
    main()
