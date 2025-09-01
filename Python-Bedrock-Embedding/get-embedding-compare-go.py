import os
import sys
import json
import time
import boto3

# Constants
DEFAULT_REGION = "us-east-1"
TITAN_EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"

def main():
    region = os.getenv("AWS_REGION", DEFAULT_REGION)
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
        modelId=TITAN_EMBEDDING_MODEL_ID,
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

if __name__ == "__main__":
    main()