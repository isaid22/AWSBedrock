## List available models in AWS Bedrock

This repo contains code that list available models in AWS Bedrock. The steps in general are as follow:

1. Create a Go module to collect related packages.

```go
go mod init example/getModels
```

2. Compile the code.

```go
go build -o getModels getModels.go
```

3. Run the compiled program.

```go
./getModels
```

expect to see:

```
Name: Pegasus v1.2 | Provider: TwelveLabs | Id: twelvelabs.pegasus-1-2-v1:0 | Modality: [TEXT]
Name: Claude Opus 4.1 | Provider: Anthropic | Id: anthropic.claude-opus-4-1-20250805-v1:0 | Modality: [TEXT]
Name: Titan Text Large | Provider: Amazon | Id: amazon.titan-tg1-large | Modality: [TEXT]
Name: Titan Image Generator G1 | Provider: Amazon | Id: amazon.titan-image-generator-v1:0 | Modality: [IMAGE]
Name: Titan Image Generator G1 | Provider: Amazon | Id: amazon.titan-image-generator-v1 | Modality: [IMAGE]
Name: Titan Image Generator G1 v2 | Provider: Amazon | Id: amazon.titan-image-generator-v2:0 | Modality: [IMAGE]
Name: Nova Premier | Provider: Amazon | Id: amazon.nova-premier-v1:0:8k | Modality: [TEXT]
Name: Nova Premier | Provider: Amazon | Id: amazon.nova-premier-v1:0:20k | Modality: [TEXT]
...
```