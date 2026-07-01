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
    def load(self):
        print("[LLM] Loading model...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )

    # -------------------------
    # UNLOAD (CRITICAL FOR KAGGLE)
    # -------------------------
    def unload(self):
        print("[LLM] Unloading model...")
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.3,
            top_p=0.9
        )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)

        json_text = self._extract_json(decoded)

        return json.loads(json_text)

    # -------------------------
    # SAFE JSON EXTRACTION
    # -------------------------
    def _extract_json(self, text: str):

        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            raise ValueError("No JSON found in LLM output")

        return match.group(0)
