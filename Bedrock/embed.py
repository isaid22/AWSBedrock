import boto3, json

# Bedrock runtime client
brt = boto3.client("bedrock-runtime", region_name="us-east-1")

# Titan embedding model ID
model_id = "amazon.titan-embed-text-v1"

# Example input text
text = "Vector databases store embeddings for semantic search."

body = {
    "inputText": text
}

resp = brt.invoke_model(
    modelId=model_id,
    body=json.dumps(body)
)

# Parse response
payload = json.loads(resp["body"].read())
embedding = payload["embedding"]

print(f"Input: {text}")
print(f"Embedding length: {len(embedding)}")
print(f"First 10 dims: {embedding[:10]}")

