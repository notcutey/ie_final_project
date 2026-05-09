# vt_siglip/text_encoder.py
import os
import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    CLIPTextModel,
    CLIPTokenizer,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False


DEFAULT_LLM_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_BERT_ID = "bert-base-uncased"
DEFAULT_CLIP_ID = "openai/clip-vit-base-patch32"


class LLMTextEncoder(nn.Module):
    """
    통합 Text Encoder
    지원:
      - encoder_type="llm"  -> AutoModelForCausalLM (예: LLaMA)
      - encoder_type="bert" -> AutoModel (예: BERT)
      - encoder_type="clip" -> CLIPTextModel
    """

    def __init__(
        self,
        model_name: str | None = None,
        encoder_type: str = "llm",   # "llm" | "bert" | "clip"
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        train_llm: bool = False,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
        pooling: str = "mean",
        hf_token: str | None = None,
    ):
        super().__init__()

        self.encoder_type = encoder_type.lower()
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.train_llm = train_llm

        token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

        if self.encoder_type == "llm":
            self.model_name = model_name or DEFAULT_LLM_ID
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                token=token,
                trust_remote_code=False,
            )
            self.lm = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
                token=token,
                trust_remote_code=False,
            )

            loaded_id = getattr(self.lm.config, "_name_or_path", "")
            print(f"[TextEncoder] type=llm | loaded={loaded_id}")

            try:
                self.tokenizer.padding_side = "left"
            except Exception:
                pass
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if self.train_llm and use_lora:
                if not _PEFT_AVAILABLE:
                    raise RuntimeError("peft 패키지가 필요함: pip install peft")
                lcfg = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=list(lora_target_modules),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                )
                self.lm = get_peft_model(self.lm, lcfg)

            if self.train_llm:
                self.lm.train()
            else:
                for p in self.lm.parameters():
                    p.requires_grad = False
                self.lm.eval()

            self.backbone = self.lm

        elif self.encoder_type == "bert":
            self.model_name = model_name or DEFAULT_BERT_ID
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                token=token,
                trust_remote_code=False,
            )
            self.backbone = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                token=token,
                trust_remote_code=False,
            )

            print(f"[TextEncoder] type=bert | loaded={self.model_name}")

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.sep_token or self.tokenizer.eos_token or "[PAD]"

            if self.train_llm:
                self.backbone.train()
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                self.backbone.eval()

        elif self.encoder_type == "clip":
            self.model_name = model_name or DEFAULT_CLIP_ID
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.model_name,
                token=token,
            )
            self.backbone = CLIPTextModel.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                token=token,
                use_safetensors=True   
            )

            print(f"[TextEncoder] type=clip | loaded={self.model_name}")

            if self.train_llm:
                self.backbone.train()
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                self.backbone.eval()

        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}")

        self.to(self.device)

    def _model_device(self) -> torch.device:
        try:
            return next(self.backbone.parameters()).device
        except StopIteration:
            return torch.device(self.device if isinstance(self.device, str) else "cpu")

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        dev = self._model_device()

        with torch.set_grad_enabled(self.train_llm):
            input_ids_dev = input_ids.to(dev, non_blocking=True)
            attention_mask_dev = attention_mask.to(dev, non_blocking=True)

            if self.encoder_type == "llm":
                out = self.backbone(
                    input_ids=input_ids_dev,
                    attention_mask=attention_mask_dev,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = out.hidden_states[-1]  # [N, L, H]

            else:
                out = self.backbone(
                    input_ids=input_ids_dev,
                    attention_mask=attention_mask_dev,
                    return_dict=True,
                )
                hidden = out.last_hidden_state  # [N, L, H]

            if self.pooling == "mean":
                mask = attention_mask_dev.unsqueeze(-1).to(hidden.dtype)
                summed = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1e-6)
                emb = summed / denom

            elif self.pooling == "cls":
                emb = hidden[:, 0, :]

            elif self.pooling == "eos":
                if self.encoder_type == "llm":
                    eos_id = self.tokenizer.eos_token_id
                elif self.encoder_type == "clip":
                    eos_id = self.tokenizer.eos_token_id
                else:
                    eos_id = self.tokenizer.sep_token_id

                ids_list = input_ids_dev.tolist()
                idxs = []
                for seq in ids_list:
                    try:
                        last_idx = len(seq) - 1 - seq[::-1].index(eos_id)
                    except ValueError:
                        last_idx = len(seq) - 1
                    idxs.append(last_idx)

                idxs = torch.tensor(idxs, device=hidden.device, dtype=torch.long)
                emb = hidden[torch.arange(hidden.size(0), device=hidden.device), idxs]

            else:
                raise ValueError(f"Unsupported pooling: {self.pooling}")

            return emb

    def detect_hidden_size(self) -> int:
        dev = self._model_device()
        tmp = self.tokenizer(
            ["hello"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        tmp = {k: v.to(dev) for k, v in tmp.items()}

        with torch.set_grad_enabled(self.train_llm):
            if self.encoder_type == "llm":
                out = self.backbone(
                    input_ids=tmp["input_ids"],
                    attention_mask=tmp["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                hidden = out.hidden_states[-1]
            else:
                out = self.backbone(
                    input_ids=tmp["input_ids"],
                    attention_mask=tmp["attention_mask"],
                    return_dict=True,
                )
                hidden = out.last_hidden_state

        return hidden.size(-1)