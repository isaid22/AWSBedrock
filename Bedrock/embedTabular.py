import pandas as pd
import json
import boto3

def format_row_for_embedding(row: pd.Series, separator: str=":", line_break: str = "\n") ->str:
    formatted_parts = [f"{col}{separator}{val}" for col, val in row.items()]
    return line_break.join(formatted_parts)



df = pd.DataFrame({
        'patient_id': [101, 102],
        'symptoms': ['cough and fever', 'headache and sore throat'],
        'diagnosis' : ['Influenza', 'common cold'],
        'treatment' : ['rest and fluids', ' over the counter meds']
})

observation_series = df.iloc[0]

text_to_embed = format_row_for_embedding(observation_series)

print(text_to_embed)


client = boto3.client(
    service_name = 'bedrock-runtime',
    region_name = 'us-east-1'
)

model_id = 'amazon.titan-embed-text-v1'
payload = {"inputText": text_to_embed}
body = json.dumps(payload)

# Invoke model
response = client.invoke_model(
    body = body,
    modelId = model_id,
    accept = 'application/json',
    contentType = 'application/json'
)

# Parse the response
response_body = json.loads(response.get('body').read())
embedding_vector = response_body.get('embedding')

# Print embedding vector
print(f"Embedding vector shape: {len(embedding_vector)}")
print(f"Preview of the vector: {embedding_vector[:5]}...")
