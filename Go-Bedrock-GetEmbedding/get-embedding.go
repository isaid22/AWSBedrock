// Reference: https://github.com/build-on-aws/amazon-bedrock-go-sdk-examples/blob/main/titan-text-embedding/main.go

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

const defaultRegion = "us-east-1"

const (
	titanEmbeddingModelID = "amazon.titan-embed-text-v1" //https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html
)

func main() {

	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = defaultRegion
	}

	cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
	if err != nil {
		log.Fatal(err)
	}

	brc := bedrockruntime.NewFromConfig(cfg)

	input := os.Args[1]

	payload := Request{
		InputText: input,
	}
	// Convert json payload to bytes array.
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		log.Fatal(err)
	}

	start := time.Now()
	output, err := brc.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		Body:        payloadBytes,
		ModelId:     aws.String(titanEmbeddingModelID),
		ContentType: aws.String("application/json"),
	})
	elapsed := time.Since(start)

	if err != nil {
		log.Fatal("failed to invoke model: ", err)
	}

	var resp Response

	startUnmarshal := time.Now()
	err = json.Unmarshal(output.Body, &resp)
	unmarshalElapsed := time.Since(startUnmarshal)

	if err != nil {
		log.Fatal("failed to unmarshal", err)
	}

	fmt.Println("embedding vector from LLM\n", resp.Embedding)
	fmt.Println()

	fmt.Println("generated embedding for input -", input)
	fmt.Println("generated vector length -", len(resp.Embedding))
	fmt.Printf("Bedrock model call took: %v\n", elapsed)
	fmt.Printf("Unmarshal took: %v\n", unmarshalElapsed)
}

//request/response model

type Request struct {
	InputText string `json:"inputText"`
}

type Response struct {
	Embedding           []float64 `json:"embedding"`
	InputTextTokenCount int       `json:"inputTextTokenCount"`
}
