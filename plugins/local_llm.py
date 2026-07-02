import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import re

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
        print("[LLM] Loading model weights into VRAM...")
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
        print("[LLM] Unloading model weights...")
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

        full_prompt = f"""
You are a strict JSON generator.

RULES:
- Output ONLY valid JSON.
- No explanation.
- No markdown.
- Must follow schema exactly.

SCHEMA:
{json.dumps(schema, indent=2)}

TASK:
{prompt}

OUTPUT:
"""

        inputs = self.tokenizer(
            full_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        print(f"[LLM] Prompt tokenized to {inputs.input_ids.shape[1]} tokens (max 2048).")

        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        )

        # Decode ONLY the newly generated tokens (strip the echoed input prompt)
        input_length = inputs.input_ids.shape[1]
        new_tokens = output[0][input_length:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"[LLM] Raw output ({len(decoded)} chars): {decoded[:120]!r}")

        json_text = self._extract_json(decoded)

        return json.loads(json_text)

    # -------------------------
    # SAFE JSON EXTRACTION
    # -------------------------
    def _extract_json(self, text: str) -> str:
        """
        Robustly extract the first syntactically complete JSON object from
        LLM output. Uses json.JSONDecoder.raw_decode so it stops exactly at
        the end of the first valid object and ignores any trailing text,
        avoiding the 'Extra data' JSONDecodeError caused by the greedy regex.
        """
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch == '{':
                try:
                    obj, _ = decoder.raw_decode(text, i)
                    return json.dumps(obj)   # re-serialise for a clean string
                except json.JSONDecodeError:
                    continue  # that '{' wasn't the start of a valid object

        raise ValueError(
            f"No valid JSON object found in LLM output. "
            f"First 200 chars: {text[:200]!r}"
        )
