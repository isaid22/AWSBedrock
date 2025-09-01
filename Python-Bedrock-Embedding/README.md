## Get Embedding using Python
This example demonstrates how to use `python` to make an API call to an embedding model hosted in AWS Bedrock.

1. Run the program.
```python
python get-embedding-compare-go.py 'Now is the time'
```

Expected results:

```
embedding vector from LLM
 [0.427734375, -0.22265625, -0.259765625, -0.01263427734375, 1.734375, 
 ...
 generated embedding for input - Now is the time
generated vector length - 1536
Bedrock model call took: 109.74 ms
Unmarshal took: 324 µs
```
