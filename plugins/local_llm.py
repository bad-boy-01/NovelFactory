import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re
import sys

from plugins.interfaces import LLMProvider


class LocalLLMProvider(LLMProvider):

    def __init__(self, model_id="Qwen/Qwen1.5-4B-Chat"):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None

    # -------------------------
    # LOAD MODEL (VRAM CONTROLLED)
    # -------------------------
    def initialize(self):
        print("[LLM] Initializing tokenizer and config (No VRAM penalty)...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        # In a real system, you might load config here too.

    def load(self):
        if self.model is not None:
            print("[Resource] Reusing resident LLM.")
            return
            
        print("[Resource] Loading LLM weights into VRAM...")
        if not self.tokenizer:
            self.initialize()
            
        try:
            import bitsandbytes
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            print("[LLM] Using bitsandbytes 4-bit quantization.")
        except Exception as e:
            bnb_config = None
            print(f"[LLM] WARNING: bitsandbytes unavailable ({e}). Falling back to fp16.")

        if bnb_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

    def unload(self):
        from core.utils.vram import flush_vram
        print("[Resource] Unloading LLM...")
        if self.model:
            del self.model
            self.model = None
        flush_vram("LLM unloaded")

    def shutdown(self):
        print("[LLM] Shutting down tokenizer...")
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

    # -------------------------
    # JSON GENERATION CORE
    # -------------------------
    def generate_json(self, prompt: str, schema: dict) -> dict:
        full_prompt = f"""You are a strict JSON generator. Output ONLY a single valid JSON object.
Do NOT output a JSON array at the top level.
Do NOT include any explanation, markdown, or code fences.
The response must start with {{ and end with }}.

SCHEMA (the exact keys your object must contain):
{json.dumps(schema, indent=2)}

TASK:
{prompt}

JSON OUTPUT:
"""

        temperature = 0.3
        
        # LLM Response Cache
        import hashlib
        from pathlib import Path
        cache_key = hashlib.sha256(f"{full_prompt}_{temperature}_{self.model_id}".encode()).hexdigest()
        cache_dir = Path("workspace/cache/llm")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            print(f"[LLM] Prompt Cache HIT ({cache_key[:8]})", flush=True)
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        n_prompt_tokens = inputs.input_ids.shape[1]
        print(f"[LLM] Generating... ({n_prompt_tokens} prompt tokens) (Cache MISS: {cache_key[:8]})", flush=True)

        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
        )

        # Decode ONLY the newly generated tokens (strip the echoed input prompt)
        input_length = inputs.input_ids.shape[1]
        new_tokens = output[0][input_length:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"[LLM] Raw output ({len(decoded)} chars): {decoded[:120]!r}", flush=True)

        json_text = self._extract_json(decoded)
        result = json.loads(json_text)
        
        # Save to LLM Cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        return result

    # -------------------------
    # SAFE JSON EXTRACTION
    # -------------------------
    def _extract_json(self, text: str) -> str:
        """
        Robustly extract the first syntactically complete JSON object from
        LLM output.

        Strategy:
          1. Try raw_decode on each '{' for strict JSON.
          2. If nothing parses, attempt _repair_json to fix common LLM
             mistakes (unquoted keys, single-quoted strings) then retry.
          3. Raise ValueError with the first 200 chars for debugging.
        """
        decoder = json.JSONDecoder()

        def _try_decode(s: str):
            for i, ch in enumerate(s):
                if ch == '{':
                    try:
                        obj, _ = decoder.raw_decode(s, i)
                        return json.dumps(obj)
                    except json.JSONDecodeError:
                        continue
            return None

        # Pass 1 — strict JSON
        result = _try_decode(text)
        if result:
            return result

        # Pass 2 — repaired JSON (handles JS-style unquoted keys, single quotes)
        repaired = self._repair_json(text)
        result = _try_decode(repaired)
        if result:
            print(f"[LLM] JSON repaired (unquoted keys or single quotes fixed)", flush=True)
            return result

        raise ValueError(
            f"No valid JSON object found in LLM output. "
            f"First 200 chars: {text[:200]!r}"
        )

    def _repair_json(self, text: str) -> str:
        """
        Best-effort fix for common LLM JSON mistakes:
        - Unquoted object keys:  { key: value }  →  { "key": value }
        - Single-quoted strings: { 'k': 'v' }   →  { "k": "v" }
        """
        # Quote unquoted keys: word characters followed by colon,
        # not already preceded by a quote or another word char.
        repaired = re.sub(r'(?<!["\w])([a-zA-Z_]\w*)\s*:', r'"\1":', text)
        # Replace single-quoted string values/keys with double quotes
        repaired = re.sub(r"'([^']*)'", r'"\1"', repaired)
        return repaired

