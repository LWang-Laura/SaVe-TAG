# src/llm_backends.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class GenConfig:
    model_path: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    device: str = "cuda"
    dtype: str = "auto"            # "auto" | "bfloat16" | "float16" | "float32"
    device_map: str = "auto"
    trust_remote_code: bool = True # Qwen often needs this

class HFChatModel:
    """
    Unified HF chat generator using the tokenizer's chat template.
    Works for Llama, Qwen, Mistral instruct models alike.
    """
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg

        dtype_map = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(cfg.dtype, "auto")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_path, use_fast=True, trust_remote_code=cfg.trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            torch_dtype=torch_dtype if torch_dtype != "auto" else None,
            device_map=cfg.device_map,
            trust_remote_code=cfg.trust_remote_code
        )

        # Some tokenizers don’t have pad token; set safely
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def chat_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        gen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Single-turn chat for a batch of user prompts.
        We rely on tokenizer.apply_chat_template for correct formatting.
        """
        outputs = []
        for user_text in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_text})

            # Convert messages -> model-specific chat format
            # add_generation_prompt=True appends the assistant token per template.
            # Ref: HF chat templating docs and Qwen chat guide.
            # (See README citations in PR)
            chat_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(
                chat_text,
                return_tensors="pt"
            ).to(self.model.device)

            gen_args = dict(
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                repetition_penalty=self.cfg.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            if gen_kwargs:
                gen_args.update(gen_kwargs)

            generated = self.model.generate(**inputs, **gen_args)
            text = self.tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            outputs.append(text.strip())
        return outputs

# Convenience factory
def build_chat_model(llm_model: str, model_path: str, **kwargs) -> HFChatModel:
    """
    llm_model: one of {"llama", "qwen", "mistral"} – used mainly for defaults.
    model_path: HF hub id or local path, e.g.
        - "meta-llama/Meta-Llama-3-8B-Instruct"
        - "Qwen/Qwen2.5-7B-Instruct"
        - "mistralai/Mistral-7B-Instruct-v0.3"
    """
    defaults = {
        "llama": dict(temperature=0.7, top_p=0.9),
        "qwen":  dict(temperature=0.7, top_p=0.9, trust_remote_code=True),
        "mistral": dict(temperature=0.7, top_p=0.95)
    }.get(llm_model.lower(), {})

    cfg = GenConfig(model_path=model_path, **{**defaults, **kwargs})
    return HFChatModel(cfg)
