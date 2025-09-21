import json
import logging
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import yaml
import time
import pandas as pd 

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
        #"stopSequences": ["."],
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

    print(response)


    # for i in range(num_requests):
    #     start_time = time.time()
    #     try:
    #         response = bedrock_client.invoke_model(
    #             modelId=model_id,
    #             contentType='application/json',
    #             accept='application/json',
    #             body=json.dumps({
    #                 "messages": messages,
    #                 "max_tokens": max_tokens,
    #                 "temperature": temperature,
    #                 "top_p": top_p
    #             })
    #         )
    #         end_time = time.time()
    #         latency = end_time - start_time
    #         latencies.append(latency)

    #         response_body = json.loads(response['body'].read().decode('utf-8'))
    #         responses.append(response_body)

    #         logger.info(f"Response {i+1}: {response_body}")
    #         logger.info(f"Latency {i+1}: {latency:.4f} seconds")

    #     except ClientError as e:
    #         logger.error(f"ClientError: {e.response['Error']['Message']}")
    #     except Exception as e:
    #         logger.error(f"Exception: {str(e)}")

    # # Save responses and latencies to a CSV file
    # df = pd.DataFrame({
    #     'response': responses,
    #     'latency_seconds': latencies
    # })
    # df.to_csv('bedrock_responses.csv', index=False)
    # logger.info("Responses and latencies saved to bedrock_responses.csv")



if __name__ == "__main__":
    main()
    