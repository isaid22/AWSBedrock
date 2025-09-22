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
import math as _math


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


def get_titan_embedding(text: str, model_id: str = "amazon.titan-embed-text-v1", region: str = "us-east-1") -> list:
	"""Return the embedding vector for the given text using Titan Embeddings."""
	client = boto3.client("bedrock-runtime", region_name=region)
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
	max_workers: int = 4,
	retries: int = 2,
	base_backoff: float = 0.5,
):
	"""Batch embed multiple texts with Titan, preserving order.

	Args:
		texts: list of strings to embed.
		model_id: Titan embedding model id.
		region: AWS region for Bedrock.
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


def _dot(a: list, b: list) -> float:
	return float(sum(x * y for x, y in zip(a, b)))


def _norm(a: list) -> float:
	return _math.sqrt(float(sum(x * x for x in a)))


def cosine_similarity(v1: list, v2: list) -> float:
	"""Compute cosine similarity between two vectors. Returns 0.0 if any norm is zero."""
	if not v1 or not v2:
		return 0.0
	den = _norm(v1) * _norm(v2)
	if den == 0:
		return 0.0
	return _dot(v1, v2) / den


def rank_by_cosine(reference_vec: list, candidate_vecs: list[list], candidate_texts: list[str]):
	"""Return candidates ranked by cosine similarity to reference_vec (desc)."""
	items = []
	for i, (vec, text) in enumerate(zip(candidate_vecs, candidate_texts)):
		sim = cosine_similarity(reference_vec, vec)
		items.append({
			"index": i,
			"message": text,
			"cosine_similarity": sim,
		})
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

	# Single embedding demo
	emb = get_titan_embedding(prompt)
	print(f"Embedding length: {len(emb)}")
	print(f"First 8 dims: {emb[:8]}")

	# Batch embedding demo: concatenated prompt + 10 candidates = 11 embeddings
	texts = [prompt] + candidate_messages
	vecs = get_titan_embeddings_batch(texts)
	print(f"Batch count: {len(vecs)}")
	print(f"Batch lens: {[len(v) for v in vecs]}")
	print(type(vecs))
	print(vecs[0])

	# Cosine similarity ranking: use the first vector (concatenated prompt) as reference
	ref_vec = vecs[0]
	cand_vecs = vecs[1:]
	ranked = rank_by_cosine(ref_vec, cand_vecs, candidate_messages)

	# Pretty-print as JSON with rounded similarities
	for item in ranked:
		item["cosine_similarity"] = round(item["cosine_similarity"], 6)
	print("\nRanked similarities (JSON):")
	print(json.dumps(ranked, indent=2))