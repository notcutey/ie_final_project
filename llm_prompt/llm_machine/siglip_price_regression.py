import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from llm_machine.text_encoder import LLMTextEncoder


PRICE_RE = re.compile(r"[\d,]+")


def parse_price_won(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    match = PRICE_RE.search(text)
    if not match:
        return None
    return float(match.group(0).replace(",", ""))


def split_final_description(text: str, drop_null_fields: bool = False) -> Tuple[List[str], List[str]]:
    fields: List[str] = []
    labels: List[str] = []

    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if drop_null_fields and value.lower() in {"null", "none", "n/a", ""}:
                continue
            labels.append(key)
            fields.append(f"{key}: {value}")
        else:
            labels.append(f"field_{len(labels)}")
            fields.append(line)

    if not fields:
        fields = ["Description: null"]
        labels = ["Description"]

    return fields, labels


def _load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_merged_items(path: str | os.PathLike) -> List[Dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected a list or dict with items list: {path}")


def load_segment_paths_by_item(segmentation_json: str | os.PathLike) -> Dict[int, List[str]]:
    data = _load_json(segmentation_json)
    groups = data.get("items", data) if isinstance(data, dict) else data
    by_item: Dict[int, List[str]] = {}

    for group in groups or []:
        for image_record in group.get("images", []):
            item_index = image_record.get("item_index")
            if item_index is None:
                continue
            paths = by_item.setdefault(int(item_index), [])
            for segment in image_record.get("segments", []):
                path = segment.get("segmented_image")
                if path and os.path.isfile(path):
                    paths.append(path)

    return by_item


class PriceRegressionDataset(Dataset):
    def __init__(
        self,
        merged_json: str | os.PathLike,
        segmentation_json: str | os.PathLike,
        image_transform=None,
        max_images_per_item: int = 4,
        drop_null_fields: bool = False,
        min_price: float = 1.0,
    ):
        self.image_transform = image_transform
        self.max_images_per_item = max(1, int(max_images_per_item))
        self.drop_null_fields = drop_null_fields

        items = load_merged_items(merged_json)
        seg_by_item = load_segment_paths_by_item(segmentation_json)

        samples: List[Dict[str, Any]] = []
        skipped_no_price = 0
        skipped_no_image = 0

        for item_index, item in enumerate(items):
            price = parse_price_won(item.get("price"))
            if price is None or price < min_price:
                skipped_no_price += 1
                continue

            image_paths = seg_by_item.get(item_index, [])
            image_paths = [p for p in image_paths if os.path.isfile(p)]
            if not image_paths:
                skipped_no_image += 1
                continue

            fields, field_names = split_final_description(
                item.get("final_description", ""),
                drop_null_fields=drop_null_fields,
            )

            samples.append(
                {
                    "item_index": item_index,
                    "title": item.get("title", f"item_{item_index}"),
                    "price_won": float(price),
                    "target_log_price": math.log1p(float(price)),
                    "image_paths": image_paths[: self.max_images_per_item],
                    "field_texts": fields,
                    "field_names": field_names,
                    "raw_item": item,
                }
            )

        self.samples = samples
        self.skipped_no_price = skipped_no_price
        self.skipped_no_image = skipped_no_image

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.image_transform is not None:
            return self.image_transform(img)
        raise ValueError("image_transform is required")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        images = [self._load_image(path) for path in sample["image_paths"]]
        return {
            "item_index": sample["item_index"],
            "title": sample["title"],
            "price_won": sample["price_won"],
            "target_log_price": sample["target_log_price"],
            "images": images,
            "image_paths": sample["image_paths"],
            "field_texts": sample["field_texts"],
            "field_names": sample["field_names"],
        }


@dataclass
class PriceBatch:
    images: torch.Tensor
    image_mask: torch.Tensor
    target_log_price: torch.Tensor
    price_won: torch.Tensor
    field_input_ids: torch.Tensor
    field_attention_mask: torch.Tensor
    field_mask: torch.Tensor
    field_names: List[List[str]]
    field_texts: List[List[str]]
    image_paths: List[List[str]]
    titles: List[str]
    item_indices: List[int]


class PriceRegressionCollator:
    def __init__(self, tokenizer, max_length: int = 128, max_fields: int = 12):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.max_fields = int(max_fields)

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> PriceBatch:
        batch_size = len(batch)
        max_images = max(len(x["images"]) for x in batch)
        c, h, w = batch[0]["images"][0].shape

        images = torch.zeros(batch_size, max_images, c, h, w, dtype=batch[0]["images"][0].dtype)
        image_mask = torch.zeros(batch_size, max_images, dtype=torch.bool)

        field_texts: List[List[str]] = []
        field_names: List[List[str]] = []
        flat_texts: List[str] = []
        field_mask = torch.zeros(batch_size, self.max_fields, dtype=torch.bool)

        for i, sample in enumerate(batch):
            for j, img in enumerate(sample["images"]):
                images[i, j] = img
                image_mask[i, j] = True

            texts = sample["field_texts"][: self.max_fields]
            names = sample["field_names"][: self.max_fields]
            field_texts.append(texts)
            field_names.append(names)

            for j in range(self.max_fields):
                if j < len(texts):
                    flat_texts.append(texts[j])
                    field_mask[i, j] = True
                else:
                    flat_texts.append("")

        encoded = self.tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        field_input_ids = encoded["input_ids"].view(batch_size, self.max_fields, -1)
        field_attention_mask = encoded["attention_mask"].view(batch_size, self.max_fields, -1)

        return PriceBatch(
            images=images,
            image_mask=image_mask,
            target_log_price=torch.tensor([x["target_log_price"] for x in batch], dtype=torch.float32),
            price_won=torch.tensor([x["price_won"] for x in batch], dtype=torch.float32),
            field_input_ids=field_input_ids,
            field_attention_mask=field_attention_mask,
            field_mask=field_mask,
            field_names=field_names,
            field_texts=field_texts,
            image_paths=[x["image_paths"] for x in batch],
            titles=[x["title"] for x in batch],
            item_indices=[int(x["item_index"]) for x in batch],
        )


class TokenAttentionExtractor(nn.Module):
    """Wrapper around networks.RetrievalNet_token_multi.Token that can expose spatial maps."""

    def __init__(self, token_model: nn.Module):
        super().__init__()
        self.token_model = token_model

    def forward(
        self,
        images: torch.Tensor,
        return_spatial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        features = self.token_model.backbone(images)
        tr = self.token_model.tr
        bsz, _, h_feat, w_feat = features.size()

        x = tr.conv(features).reshape(bsz, tr.mid_dim, h_feat * w_feat).permute(0, 2, 1)
        for encoder in tr.encoder:
            x = encoder(x)

        q = tr.query.repeat(bsz, 1, 1)
        query_attn = F.softmax(torch.bmm(q, x.permute(0, 2, 1)), dim=1)
        token = torch.bmm(query_attn, x)
        token = tr.token_norm(token)

        decoder_attn = None
        for decoder in tr.decoder:
            token, decoder_attn = decoder(token, x)

        token = F.normalize(token, dim=-1)
        out = {"tokens": token}

        if return_spatial:
            out["query_attention"] = query_attn.view(bsz, tr.query.size(1), h_feat, w_feat)
            if decoder_attn is not None:
                out["decoder_attention"] = decoder_attn.view(
                    bsz,
                    decoder_attn.size(1),
                    decoder_attn.size(2),
                    h_feat,
                    w_feat,
                )
        return out


class PriceRegressionAttentionModel(nn.Module):
    def __init__(
        self,
        text_encoder: LLMTextEncoder,
        vision_dim: int = 1024,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        huber_beta: float = 0.15,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = getattr(text_encoder, "tokenizer", None)
        llm_hidden_size = text_encoder.detect_hidden_size()

        self.text_proj = nn.Sequential(
            nn.Linear(llm_hidden_size, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        if vision_dim == hidden_dim:
            self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        else:
            self.vision_proj = nn.Sequential(
                nn.LayerNorm(vision_dim),
                nn.Linear(vision_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        self.token_score = nn.Linear(hidden_dim, 1)
        self.image_score = nn.Linear(hidden_dim, 1)
        self.field_score = nn.Linear(hidden_dim, 1)
        self.modality_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.huber_beta = float(huber_beta)

    def encode_fields(
        self,
        field_input_ids: torch.Tensor,
        field_attention_mask: torch.Tensor,
        field_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_fields, seq_len = field_input_ids.shape
        flat_ids = field_input_ids.reshape(bsz * num_fields, seq_len)
        flat_mask = field_attention_mask.reshape(bsz * num_fields, seq_len)

        text_raw = self.text_encoder.encode_text(flat_ids, flat_mask)
        text_raw = text_raw.to(next(self.text_proj.parameters()).dtype)
        text_emb = self.text_proj(text_raw).reshape(bsz, num_fields, -1)
        text_emb = text_emb * field_mask.unsqueeze(-1).to(text_emb.dtype)
        return text_emb

    def aggregate_image_tokens(
        self,
        image_tokens: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # image_tokens: [B, Nimg, Ntok, Dv]
        bsz, num_images, num_tokens, _ = image_tokens.shape
        v = self.vision_proj(image_tokens.reshape(bsz * num_images * num_tokens, -1))
        v = v.reshape(bsz, num_images, num_tokens, -1)

        token_logits = self.token_score(v).squeeze(-1)
        invalid_images = ~image_mask
        token_logits = token_logits.masked_fill(invalid_images.unsqueeze(-1), -1e4)
        token_attn = torch.softmax(token_logits, dim=-1)
        image_emb = (v * token_attn.unsqueeze(-1)).sum(dim=2)

        image_logits = self.image_score(image_emb).squeeze(-1)
        image_logits = image_logits.masked_fill(invalid_images, -1e4)
        image_attn = torch.softmax(image_logits, dim=-1)
        visual_emb = (image_emb * image_attn.unsqueeze(-1)).sum(dim=1)
        return visual_emb, image_emb, image_attn, token_attn

    def aggregate_fields(
        self,
        text_emb: torch.Tensor,
        field_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        field_logits = self.field_score(text_emb).squeeze(-1)
        field_logits = field_logits.masked_fill(~field_mask, -1e4)
        field_attn = torch.softmax(field_logits, dim=-1)
        text_summary = (text_emb * field_attn.unsqueeze(-1)).sum(dim=1)
        return text_summary, field_attn

    def forward(
        self,
        image_tokens: torch.Tensor,
        image_mask: torch.Tensor,
        field_input_ids: torch.Tensor,
        field_attention_mask: torch.Tensor,
        field_mask: torch.Tensor,
        target_log_price: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        device = next(self.parameters()).device
        image_tokens = image_tokens.to(device, non_blocking=True)
        image_mask = image_mask.to(device, non_blocking=True)
        field_input_ids = field_input_ids.to(device, non_blocking=True)
        field_attention_mask = field_attention_mask.to(device, non_blocking=True)
        field_mask = field_mask.to(device, non_blocking=True)

        visual_emb, image_emb, image_attn, token_attn = self.aggregate_image_tokens(image_tokens, image_mask)
        text_emb = self.encode_fields(field_input_ids, field_attention_mask, field_mask)
        text_summary, field_attn = self.aggregate_fields(text_emb, field_mask)

        gate = torch.sigmoid(self.modality_gate(torch.cat([visual_emb, text_summary], dim=-1)))
        fused = (gate * visual_emb) + ((1.0 - gate) * text_summary)
        pred_log_price = self.regressor(torch.cat([visual_emb, text_summary, fused], dim=-1)).squeeze(-1)

        out: Dict[str, Any] = {
            "pred_log_price": pred_log_price,
            "pred_price_won": torch.expm1(pred_log_price).clamp(min=0.0),
            "modality_gate_image": gate.squeeze(-1),
        }

        if target_log_price is not None:
            target_log_price = target_log_price.to(device, non_blocking=True).float()
            loss = F.smooth_l1_loss(
                pred_log_price,
                target_log_price,
                beta=self.huber_beta,
                reduction="mean",
            )
            out["loss"] = loss

        if return_attention:
            out.update(
                {
                    "image_attention": image_attn,
                    "token_attention": token_attn,
                    "field_attention": field_attn,
                    "image_embeddings": image_emb,
                    "field_embeddings": text_emb,
                }
            )

        return out


def regression_metrics_from_log(pred_log: torch.Tensor, target_log: torch.Tensor) -> Dict[str, float]:
    pred_price = torch.expm1(pred_log.detach()).clamp(min=0.0)
    target_price = torch.expm1(target_log.detach()).clamp(min=0.0)
    mae = (pred_price - target_price).abs().mean()
    rmse = torch.sqrt(((pred_price - target_price) ** 2).mean())
    mape = ((pred_price - target_price).abs() / target_price.clamp(min=1.0)).mean() * 100.0
    log_mae = (pred_log.detach() - target_log.detach()).abs().mean()
    return {
        "mae_won": float(mae.cpu().item()),
        "rmse_won": float(rmse.cpu().item()),
        "mape_percent": float(mape.cpu().item()),
        "log_mae": float(log_mae.cpu().item()),
    }


def save_json(path: str | os.PathLike, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
