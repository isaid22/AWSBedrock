import boto3

region = "us-east-1"  # change if needed
model_id = "amazon.nova-lite-v1:0"  # or pro/micro

brt = boto3.client("bedrock-runtime", region_name=region)

resp = brt.converse(
    modelId=model_id,
    messages=[{"role": "user", "content": [{"text": "Give me 3 fun facts about Chicago."}]}],
    inferenceConfig={
        "maxTokens": 300,
        "temperature": 0.4,
        "topP": 0.9
    }
)

# Print the text response
for item in resp["output"]["message"]["content"]:
    if "text" in item:
        print(item["text"])
