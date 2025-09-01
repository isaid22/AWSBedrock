## Get embedding using Python and parameters

This example demonstrates how to parameterize the program to read from a configuration file for calling AWS Bedrock API for embedding. The staps are as follow:

```python
python get-embedding-compare-go-parameterized.py 
```

expect results such as:

```
embedding vector from LLM
 [0.5546875, -0.1552734375, 0.53125, 0.455078125, 0.44140625, -0.01263427734375,
 ...
 generated embedding for input - Discover how we've streamlined homebuying. We've made it simpler and easier to get your next home.
generated vector length - 1536
Bedrock model call took: 108.12 ms
Unmarshal took: 354 µs
```