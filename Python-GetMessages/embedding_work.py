import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import yaml
import boto3
from botocore.exceptions import ClientError
import time as _time
import random as _random
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn.functional as F


def _load_yaml_allow_tabs(file_path: str | Path) -> dict:
	"""Load YAML but first replace tab indentation with spaces for safety.

	YAML spec disallows tabs for indentation; this normalizes them so the
	existing run_config.yaml (which uses tabs) can still be parsed.
	"""
	p = Path(file_path)
	text = p.read_text(encoding="utf-8")
	# Normalize tabs to 2 spaces to keep list indentation reasonable
	text = text.replace("\t", "  ")
	return yaml.safe_load(text) or {}


def get_prompts_from_config(cfg: dict, system_name: str = "system_prompt", user_name: str = "user_prompt_for_home_equity") -> Tuple[Optional[str], Optional[str]]:
	"""Extract system and user (home equity) prompt templates from parsed config.

	Supports key names: 'prompt' or 'prompts' as a list of {name, template}.
	Returns (system_template, user_template), where any item can be None if missing.
	"""
	prompts = cfg.get("prompt") or cfg.get("prompts") or []
	system_t = None
	user_t = None
	if isinstance(prompts, list):
		for item in prompts:
			if not isinstance(item, dict):
				continue
			name = item.get("name")
			template = item.get("template")
			if name == system_name:
				system_t = template
			if name == user_name:
				user_t = template
	elif isinstance(prompts, dict):
		# If someone used a dict mapping names->template
		system_t = prompts.get(system_name)
		user_t = prompts.get(user_name)

	return system_t, user_t


def build_concatenated_prompt_from_yaml(config_path: str | Path,
										system_name: str = "system_prompt",
										user_name: str = "user_prompt_for_home_equity",
										separator: str = "\n\n") -> str:
	"""Read run_config.yaml, pick system + home equity user prompt, and concatenate.

	Returns a single string. Missing parts are skipped gracefully.
	"""
	cfg = _load_yaml_allow_tabs(config_path)
	system_t, user_t = get_prompts_from_config(cfg, system_name, user_name)
	parts = [p for p in (system_t, user_t) if isinstance(p, str) and p.strip()]
	return separator.join(parts)


def get_titan_embedding(text: str, model_id: str = "amazon.titan-embed-text-v1", region: str = "us-east-1", dimensions: int = 1536) -> list:
	"""Return the embedding vector for the given text using Titan Embeddings.
	
	Args:
		text: Input text to embed.
		model_id: Titan embedding model id.
		region: AWS region for Bedrock.
		dimensions: Output dimension size. Supported values depend on model:
			- amazon.titan-embed-text-v1: 1536 (fixed)
			- amazon.titan-embed-text-v2:0: 256, 512, 1024 (default: 1024)
	
	Returns:
		List of float values representing the embedding vector.
	"""
	client = boto3.client("bedrock-runtime", region_name=region)
	
	# Build request body based on model capabilities
	if "v2" in model_id:
		# Titan v2 supports configurable dimensions
		body = json.dumps({
			"inputText": text,
			"dimensions": dimensions
		})
	else:
		# Titan v1 has fixed 1536 dimensions
		if dimensions != 1536:
			print(f"Warning: Titan v1 only supports 1536 dimensions, ignoring dimensions={dimensions}")
		body = json.dumps({"inputText": text})
	
	resp = client.invoke_model(
		modelId=model_id,
		body=body,
		accept="application/json",
		contentType="application/json",
	)
	payload = json.loads(resp["body"].read())
	return payload.get("embedding", [])


def get_titan_embeddings_batch(
	texts: list,
	model_id: str = "amazon.titan-embed-text-v1",
	region: str = "us-east-1",
	dimensions: int = 1536,
	max_workers: int = 4,
	retries: int = 2,
	base_backoff: float = 0.5,
):
	"""Batch embed multiple texts with Titan, preserving order.

	Args:
		texts: list of strings to embed.
		model_id: Titan embedding model id.
		region: AWS region for Bedrock.
		dimensions: Output dimension size. Supported values depend on model:
			- amazon.titan-embed-text-v1: 1536 (fixed)
			- amazon.titan-embed-text-v2:0: 256, 512, 1024 (default: 1024)
		max_workers: parallel requests.
		retries: retry attempts per item on throttling/transient errors.
		base_backoff: seconds for exponential backoff base.

	Returns:
		List of embedding vectors (list[float]) aligned with input order.
	"""
	if not isinstance(texts, list):
		raise TypeError("texts must be a list of strings")

	client = boto3.client("bedrock-runtime", region_name=region)
	results: list = [None] * len(texts)

	def _embed_one(idx: int, text: str):
		if not isinstance(text, str):
			raise TypeError(f"texts[{idx}] is not a string")
		attempt = 0
		while True:
			try:
				# Build request body based on model capabilities
				if "v2" in model_id:
					# Titan v2 supports configurable dimensions
					body = json.dumps({
						"inputText": text,
						"dimensions": dimensions
					})
				else:
					# Titan v1 has fixed 1536 dimensions
					if dimensions != 1536 and idx == 0:  # Only warn once
						print(f"Warning: Titan v1 only supports 1536 dimensions, ignoring dimensions={dimensions}")
					body = json.dumps({"inputText": text})
				
				resp = client.invoke_model(
					modelId=model_id,
					body=body,
					accept="application/json",
					contentType="application/json",
				)
				payload = json.loads(resp["body"].read())
				return idx, payload.get("embedding", [])
			except ClientError as e:
				code = e.response.get("Error", {}).get("Code")
				status = (e.response.get("ResponseMetadata", {}) or {}).get("HTTPStatusCode")
				if code in ("ThrottlingException", "TooManyRequestsException") or status in (429, 503, 500):
					if attempt >= retries:
						raise
					delay = base_backoff * (2 ** attempt) + _random.uniform(0, 0.25)
					_time.sleep(delay)
					attempt += 1
					continue
				raise
			except Exception:
				if attempt >= retries:
					raise
				delay = base_backoff * (2 ** attempt) + _random.uniform(0, 0.25)
				_time.sleep(delay)
				attempt += 1

	with ThreadPoolExecutor(max_workers=max_workers) as ex:
		futures = [ex.submit(_embed_one, i, t) for i, t in enumerate(texts)]
		for fut in as_completed(futures):
			idx, vec = fut.result()
			results[idx] = vec

	return results


def cosine_similarity(v1: list, v2: list) -> float:
	"""Compute cosine similarity between two vectors using PyTorch. Returns 0.0 if any norm is zero."""
	if not v1 or not v2:
		return 0.0
	
	# Convert lists to PyTorch tensors
	tensor1 = torch.tensor(v1, dtype=torch.float32)
	tensor2 = torch.tensor(v2, dtype=torch.float32)
	
	# Use PyTorch's cosine similarity function
	# F.cosine_similarity returns a tensor, so we convert to float
	similarity = F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0), dim=1)
	return float(similarity.item())


def rank_by_cosine(reference_vec: list, candidate_vecs: list[list], candidate_texts: list[str]):
	"""Return candidates ranked by cosine similarity to reference_vec (desc).
	
	Uses PyTorch for efficient batch computation of cosine similarities.
	"""
	if not candidate_vecs or not candidate_texts:
		return []
	
	# Convert to PyTorch tensors for batch processing
	ref_tensor = torch.tensor(reference_vec, dtype=torch.float32).unsqueeze(0)  # Shape: (1, dim)
	candidate_tensor = torch.stack([torch.tensor(vec, dtype=torch.float32) for vec in candidate_vecs])  # Shape: (n, dim)
	
	# Compute cosine similarities for all candidates at once
	similarities = F.cosine_similarity(ref_tensor, candidate_tensor, dim=1)
	
	# Create result items with similarity scores
	items = []
	for i, (sim, text) in enumerate(zip(similarities.tolist(), candidate_texts)):
		items.append({
			"index": i,
			"message": text,
			"cosine_similarity": sim,
		})
	
	# Sort by similarity in descending order
	items.sort(key=lambda x: x["cosine_similarity"], reverse=True)
	return items


if __name__ == "__main__":
	# Quick demo when running this file directly
	candidate_messages = [
		"Unlock your home's potential with a flexible loan.",
		"Transform your home into a financial opportunity.",
		"Access your home equity for your dreams.",
		"Empower your financial future with your home.",
		"Turn your home into a resource for your goals.",
		"Leverage your home's value for your needs.",
		"Elevate your lifestyle with your home equity.",
		"Access funds to enhance your home and life.",
		"Turn your home equity into a financial advantage.",
		"Secure funds from your home's equity today.",
	]

	config_file = Path(__file__).with_name("run_config.yaml")
	prompt = build_concatenated_prompt_from_yaml(config_file)
	print(prompt)
	print(type(prompt))

	# Test different embedding dimensions and models
	test_configs = [
		# {"model_id": "amazon.titan-embed-text-v1", "dimensions": 1536, "name": "Titan v1 (1536)"},
		# Uncomment these if you have access to Titan v2:
		{"model_id": "amazon.titan-embed-text-v2:0", "dimensions": 256, "name": "Titan v2 (256)"},
		# {"model_id": "amazon.titan-embed-text-v2:0", "dimensions": 512, "name": "Titan v2 (512)"},
		# {"model_id": "amazon.titan-embed-text-v2:0", "dimensions": 1024, "name": "Titan v2 (1024)"},
	]
	
	for config in test_configs:
		print(f"\n{'='*60}")
		print(f"Testing {config['name']}")
		print(f"{'='*60}")
		
		try:
			# Single embedding demo
			emb = get_titan_embedding(prompt, model_id=config["model_id"], dimensions=config["dimensions"])
			print(f"Embedding length: {len(emb)}")
			print(f"First 8 dims: {emb[:8]}")

			# Batch embedding demo: concatenated prompt + 10 candidates
			texts = [prompt] + candidate_messages
			vecs = get_titan_embeddings_batch(texts, model_id=config["model_id"], dimensions=config["dimensions"])
			print(f"Batch count: {len(vecs)}")
			print(f"Batch lens: {[len(v) for v in vecs]}")

			# Cosine similarity ranking: use the first vector (concatenated prompt) as reference
			ref_vec = vecs[0]
			cand_vecs = vecs[1:]
			ranked = rank_by_cosine(ref_vec, cand_vecs, candidate_messages)

			# Pretty-print as JSON with rounded similarities
			for item in ranked:
				item["cosine_similarity"] = round(item["cosine_similarity"], 6)
			print(f"\nTop 3 ranked similarities for {config['name']}:")
			for i, item in enumerate(ranked[:3]):
				print(f"  {i+1}. {item['message'][:50]}... (sim: {item['cosine_similarity']})")
				
		except Exception as e:
			print(f"Error with {config['name']}: {e}")
			continue