"""
test_ollama.py – Health and capability verification script for local Ollama service.

Verifies:
  1. Ollama service responsiveness on http://127.0.0.1:11434
  2. Model catalog (/api/tags) and confirms required models exist:
     - qwen2.5:7b-instruct (LLM auditor)
     - nomic-embed-text    (ChromaDB vector embedder)
  3. Live text embedding test via /api/embed (checks llama-server backend)
  4. Live chat completion test via /api/chat (checks LLM generation)

Usage:
  python test_ollama.py
"""

import json
import os
import sys
import time

# Ensure localhost requests do not route through environment proxies
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

try:
    import requests
    session = requests.Session()
    session.trust_env = False
except ImportError:
    print("X 'requests' not installed. Install via pip install requests")
    sys.exit(1)


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:7b-instruct")
EMBED_MODEL = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


def print_step(name: str):
    print("\n" + "=" * 65)
    print(f"  {name}")
    print("=" * 65)


def check_liveness() -> bool:
    print_step("STEP 1: Checking Ollama Server Liveness")
    try:
        t0 = time.time()
        resp = session.get(f"{OLLAMA_BASE_URL}/", timeout=5)
        elapsed = round((time.time() - t0) * 1000, 1)
        if resp.status_code == 200:
            print(f"[PASS] Ollama server is ONLINE at {OLLAMA_BASE_URL} (response time: {elapsed}ms)")
            print(f"       Server response: '{resp.text.strip()}'")
            return True
        else:
            print(f"[FAIL] Ollama server returned unexpected status code: {resp.status_code}")
            return False
    except requests.exceptions.RequestException as exc:
        print(f"[FAIL] Could not connect to Ollama server at {OLLAMA_BASE_URL}: {exc}")
        print("       Make sure 'ollama serve' is running.")
        return False


def check_models() -> bool:
    print_step("STEP 2: Inspecting Available Models")
    try:
        resp = session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            print(f"[FAIL] Failed to query /api/tags: {resp.status_code}")
            return False
        
        data = resp.json()
        models = [m.get("name") for m in data.get("models", [])]
        print(f"Found {len(models)} installed models:")
        for m in models:
            print(f"   * {m}")
        
        has_chat = any(CHAT_MODEL in m for m in models)
        has_embed = any(EMBED_MODEL in m for m in models)

        print(f"\n   Checking required models:")
        print(f"   - Chat Model ({CHAT_MODEL}): {'[PASS] FOUND' if has_chat else '[FAIL] MISSING'}")
        print(f"   - Embed Model ({EMBED_MODEL}): {'[PASS] FOUND' if has_embed else '[FAIL] MISSING'}")
        
        return has_chat and has_embed
    except Exception as exc:
        print(f"[FAIL] Error checking models: {exc}")
        return False


def test_embedding() -> bool:
    print_step("STEP 3: Testing Embedding Generation (Backend llama-server)")
    test_text = "FTC compliance rule: video endorsements require clear and conspicuous disclosures."
    payload = {
        "model": EMBED_MODEL,
        "input": test_text
    }
    try:
        t0 = time.time()
        resp = session.post(f"{OLLAMA_BASE_URL}/api/embed", json=payload, timeout=30)
        elapsed = round(time.time() - t0, 2)
        if resp.status_code == 200:
            data = resp.json()
            embeddings = data.get("embeddings", [])
            if embeddings and len(embeddings) > 0:
                dim = len(embeddings[0])
                print(f"[PASS] Embedding generated successfully in {elapsed}s!")
                print(f"       Input text: '{test_text[:50]}...'")
                print(f"       Vector dimensions: {dim} (expected 768 for nomic-embed-text)")
                print(f"       Vector sample: [{', '.join(f'{x:.4f}' for x in embeddings[0][:5])}...]")
                return True
            else:
                print(f"[FAIL] No embeddings returned in response: {data}")
                return False
        else:
            print(f"[FAIL] Embedding API failed with HTTP {resp.status_code}:")
            print(f"       {resp.text}")
            return False
    except Exception as exc:
        print(f"[FAIL] Embedding test exception: {exc}")
        return False


def test_chat_completion() -> bool:
    print_step("STEP 4: Testing Chat Inference Generation (LLM Auditor)")
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise compliance auditor."},
            {"role": "user", "content": "In one short sentence, define what an FTC material connection is."}
        ],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    try:
        t0 = time.time()
        resp = session.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=60)
        elapsed = round(time.time() - t0, 2)
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            print(f"[PASS] LLM generation successful in {elapsed}s!")
            print(f"       Model: {CHAT_MODEL}")
            print(f"       Response:\n       \"{content}\"")
            return True
        else:
            print(f"[FAIL] Chat API failed with HTTP {resp.status_code}:")
            print(f"       {resp.text}")
            return False
    except Exception as exc:
        print(f"[FAIL] Chat test exception: {exc}")
        return False


def main():
    print("=" * 65)
    print("      OLLAMA COMPREHENSIVE HEALTH & READINESS CHECK")
    print("=" * 65)
    print(f"Target URL   : {OLLAMA_BASE_URL}")
    print(f"Chat Model   : {CHAT_MODEL}")
    print(f"Embed Model  : {EMBED_MODEL}")

    s1 = check_liveness()
    if not s1:
        print("\n[FAIL] Ollama verification FAILED at Step 1.")
        sys.exit(1)

    s2 = check_models()
    if not s2:
        print("\n[FAIL] Ollama verification FAILED at Step 2.")
        sys.exit(1)

    s3 = test_embedding()
    if not s3:
        print("\n[FAIL] Ollama verification FAILED at Step 3 (Embedding backend).")
        sys.exit(1)

    s4 = test_chat_completion()
    if not s4:
        print("\n[FAIL] Ollama verification FAILED at Step 4 (Chat inference backend).")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("SUCCESS: ALL CHECKS PASSED: Ollama is 100% operational and ready for RAG!")
    print("=" * 65)
    sys.exit(0)


if __name__ == "__main__":
    main()

