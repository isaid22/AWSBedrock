## Get embedding using Go and parameters

This example demonstrates how to parameterize the program to read from a configuration file for calling AWS Bedrock API for embedding. The steps are as follow:

1. Ensure `PATH` includes `Go` program

```
export PATH=$PATH:/usr/local/go/bin
```

2. Build the module.

```go
go mod init example/get-embedding-param
```
```go
go mod tidy
```

3. Build the executable.

```go
go build -o get-embedding-parameterized get-embedding-parameterized.go
```

4. Run the program.
```go
./get-embedding-parameterized
```

Expect this result:

```
embedding vector from LLM
 [0.5546875 -0.1552734375 0.53125 0.455078125 0.44140625 -0.01263427734375
 ...

 generated embedding for input - Discover how we've streamlined homebuying. We've made it simpler and easier to get your next home.
generated vector length - 1536
Bedrock model call took: 114.527869ms
Unmarshal took: 535.169µs
```