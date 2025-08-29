import json, boto3

brt = boto3.client("bedrock-runtime", region_name="us-east-1")

body = {
  "messages": [
    {"role": "user", "content": [{"text": "Explain vector databases in one paragraph."}]}
  ],
  "inferenceConfig": {"maxTokens": 300, "temperature": 0.3, "topP": 0.9}
}

resp = brt.invoke_model(
    modelId="amazon.nova-micro-v1:0",
    body=json.dumps(body)
)

payload = json.loads(resp["body"].read())
# Nova returns a message structure in "output" → "message" → "content"
for part in payload["output"]["message"]["content"]:
    if "text" in part:
        print(part["text"])
