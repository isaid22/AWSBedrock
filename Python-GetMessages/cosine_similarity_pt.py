import torch
import torch.nn.functional as F

# Assume your embeddings are 11 vectors of dimension D
D = 128 # Example dimension
embeddings = torch.randn(11, D) 

# The first embedding is the reference vector
reference_vector = embeddings[0]

# The other ten vectors
other_vectors = embeddings[1:]

# Calculate cosine similarity
# F.cosine_similarity expects inputs of the same shape for element-wise comparison,
# or it can broadcast if dimensions are compatible.
# Here, we expand the reference_vector to match the batch dimension of other_vectors.
cosine_similarities = F.cosine_similarity(reference_vector.unsqueeze(0), other_vectors, dim=1)

print("Reference vector shape:", reference_vector.shape)
print("Other vectors shape:", other_vectors.shape)
print("Cosine similarities:", cosine_similarities)
print("Number of similarities:", len(cosine_similarities))
