package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"gopkg.in/yaml.v3"
)

type Config struct {
	Region          string `yaml:"region"`
	ModelID         string `yaml:"model_id"`
	EmbeddingLength int    `yaml:"embedding_length"`
	InputText       string `yaml:"input_text"`
}

func loadConfig(path string) (*Config, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

type Request struct {
	InputText string `json:"inputText"`
}

type Response struct {
	Embedding           []float64 `json:"embedding"`
	InputTextTokenCount int       `json:"inputTextTokenCount"`
}

func main() {
	cfg, err := loadConfig("config.yaml")
	if err != nil {
		log.Fatal("failed to load config: ", err)
	}

	awsCfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(cfg.Region))
	if err != nil {
		log.Fatal(err)
	}

	brc := bedrockruntime.NewFromConfig(awsCfg)

	payload := Request{
		InputText: cfg.InputText,
	}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		log.Fatal(err)
	}

	start := time.Now()
	output, err := brc.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		Body:        payloadBytes,
		ModelId:     aws.String(cfg.ModelID),
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
	fmt.Println("generated embedding for input -", cfg.InputText)
	fmt.Println("generated vector length -", len(resp.Embedding))
	fmt.Printf("Bedrock model call took: %v\n", elapsed)
	fmt.Printf("Unmarshal took: %v\n", unmarshalElapsed)

	if cfg.EmbeddingLength > 0 && len(resp.Embedding) != cfg.EmbeddingLength {
		fmt.Printf("Warning: embedding length (%d) does not match expected (%d)\n", len(resp.Embedding), cfg.EmbeddingLength)
	}
}
