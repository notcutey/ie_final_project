#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from llm_machine.siglip_price_regression import (
    load_merged_items,
    load_segment_paths_by_item,
    parse_price_won,
    regression_metrics_from_log,
    save_json,
    split_final_description,
)
from llm_machine.text_encoder import LLMTextEncoder


PROJECT_DIR = Path("/home/policelab_l40s/llm_prompt/llm_prompt/project")
DEFAULT_MERGED_JSON = PROJECT_DIR / "merged_with_final_description.json"
DEFAULT_SEG_JSON = PROJECT_DIR / "images" / "segmentation_labels.json"
DEFAULT_CKPT_DIR = Path(
    "/home/policelab_l40s/llm_prompt/llm_prompt/llm_machine/checkpoint_price_regression_text_only"
)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    return torch.bfloat16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a text-only price regression ablation with final_description fields."
    )
    parser.add_argument("--merged-json", default=str(DEFAULT_MERGED_JSON))
    parser.add_argument("--segmentation-json", default=str(DEFAULT_SEG_JSON))
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images-per-item", type=int, default=4)
    parser.add_argument("--max-fields", type=int, default=12)
    parser.add_argument("--max-text-length", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--huber-beta", type=float, default=0.15)
    parser.add_argument("--text-encoder-type", default="llm", choices=["llm", "bert", "clip"])
    parser.add_argument("--text-model-name", default=None)
    parser.add_argument("--text-pooling", default="mean", choices=["mean", "cls", "eos"])
    parser.add_argument("--text-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--train-text-encoder", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--drop-null-fields", action="store_true")
    parser.add_argument("--regression-ckpt-path", default=None)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--explain-count", type=int, default=8)
    parser.add_argument("--explain-dir", default=None)
    return parser.parse_args()


class TextOnlyPriceRegressionDataset(Dataset):
    """Same item filtering as the multimodal dataset, but without loading image tensors."""

    def __init__(
        self,
        merged_json: str | os.PathLike,
        segmentation_json: str | os.PathLike,
        max_images_per_item: int = 4,
        drop_null_fields: bool = False,
        min_price: float = 1.0,
    ):
        self.max_images_per_item = max(1, int(max_images_per_item))
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

            image_paths = [p for p in seg_by_item.get(item_index, []) if os.path.isfile(p)]
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
                }
            )

        self.samples = samples
        self.skipped_no_price = skipped_no_price
        self.skipped_no_image = skipped_no_image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


@dataclass
class TextPriceBatch:
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


class TextOnlyCollator:
    def __init__(self, tokenizer, max_length: int = 128, max_fields: int = 12):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.max_fields = int(max_fields)

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> TextPriceBatch:
        batch_size = len(batch)
        flat_texts: List[str] = []
        field_texts: List[List[str]] = []
        field_names: List[List[str]] = []
        field_mask = torch.zeros(batch_size, self.max_fields, dtype=torch.bool)

        for i, sample in enumerate(batch):
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

        return TextPriceBatch(
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


class TextOnlyPriceRegressionModel(nn.Module):
    def __init__(
        self,
        text_encoder: LLMTextEncoder,
        hidden_dim: int = 1024,
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
        self.field_score = nn.Linear(hidden_dim, 1)
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
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
        field_input_ids: torch.Tensor,
        field_attention_mask: torch.Tensor,
        field_mask: torch.Tensor,
        target_log_price: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        device = next(self.parameters()).device
        field_input_ids = field_input_ids.to(device, non_blocking=True)
        field_attention_mask = field_attention_mask.to(device, non_blocking=True)
        field_mask = field_mask.to(device, non_blocking=True)

        text_emb = self.encode_fields(field_input_ids, field_attention_mask, field_mask)
        text_summary, field_attn = self.aggregate_fields(text_emb, field_mask)
        pred_log_price = self.regressor(text_summary).squeeze(-1)

        out: Dict[str, Any] = {
            "pred_log_price": pred_log_price,
            "pred_price_won": torch.expm1(pred_log_price).clamp(min=0.0),
        }

        if target_log_price is not None:
            target_log_price = target_log_price.to(device, non_blocking=True).float()
            out["loss"] = F.smooth_l1_loss(
                pred_log_price,
                target_log_price,
                beta=self.huber_beta,
                reduction="mean",
            )

        if return_attention:
            out.update(
                {
                    "field_attention": field_attn,
                    "field_embeddings": text_emb,
                }
            )

        return out


def build_model(args, device: torch.device):
    text_encoder = LLMTextEncoder(
        model_name=args.text_model_name,
        encoder_type=args.text_encoder_type,
        device=str(device),
        dtype=dtype_from_name(args.text_dtype),
        train_llm=args.train_text_encoder,
        use_lora=args.use_lora,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        pooling=args.text_pooling,
    )
    model = TextOnlyPriceRegressionModel(
        text_encoder=text_encoder,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        huber_beta=args.huber_beta,
    ).to(device)

    if args.regression_ckpt_path:
        raw = torch.load(args.regression_ckpt_path, map_location="cpu", weights_only=False)
        if "text_model" in raw:
            model.load_state_dict(raw["text_model"], strict=False)
            print(f"[Resume] text model <- {args.regression_ckpt_path}")

    return model


def save_checkpoint(
    path: Path,
    model: TextOnlyPriceRegressionModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args: argparse.Namespace,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "text_model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )
    print(f"[Save] {path}")


def train_one_epoch(
    model: TextOnlyPriceRegressionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
    epoch: int,
):
    model.train()
    if not args.train_text_encoder:
        model.text_encoder.eval()

    losses = []
    pred_logs = []
    target_logs = []
    t0 = time.perf_counter()

    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        out = model(
            field_input_ids=batch.field_input_ids,
            field_attention_mask=batch.field_attention_mask,
            field_mask=batch.field_mask,
            target_log_price=batch.target_log_price,
            return_attention=False,
        )
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=1.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        pred_logs.append(out["pred_log_price"].detach().cpu())
        target_logs.append(batch.target_log_price.detach().cpu())

        if step % args.log_every == 0:
            metrics = regression_metrics_from_log(torch.cat(pred_logs), torch.cat(target_logs))
            print(
                f"[Train:text] epoch={epoch} step={step}/{len(loader)} "
                f"loss={np.mean(losses):.4f} "
                f"MAE={metrics['mae_won']:,.0f}won "
                f"MAPE={metrics['mape_percent']:.2f}% "
                f"time={time.perf_counter() - t0:.1f}s"
            )

    metrics = regression_metrics_from_log(torch.cat(pred_logs), torch.cat(target_logs))
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


@torch.no_grad()
def evaluate(model: TextOnlyPriceRegressionModel, loader: DataLoader, device: torch.device):
    model.eval()
    losses = []
    pred_logs = []
    target_logs = []

    for batch in tqdm(loader, desc="Eval:text", leave=False):
        out = model(
            field_input_ids=batch.field_input_ids,
            field_attention_mask=batch.field_attention_mask,
            field_mask=batch.field_mask,
            target_log_price=batch.target_log_price,
            return_attention=False,
        )
        losses.append(float(out["loss"].cpu().item()))
        pred_logs.append(out["pred_log_price"].cpu())
        target_logs.append(batch.target_log_price.cpu())

    metrics = regression_metrics_from_log(torch.cat(pred_logs), torch.cat(target_logs))
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


@torch.no_grad()
def save_explanations(
    model: TextOnlyPriceRegressionModel,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    max_items: int,
):
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    explanations = []
    saved = 0

    for batch in loader:
        out = model(
            field_input_ids=batch.field_input_ids,
            field_attention_mask=batch.field_attention_mask,
            field_mask=batch.field_mask,
            target_log_price=batch.target_log_price,
            return_attention=True,
        )

        field_attn = out["field_attention"].cpu()
        pred_price = out["pred_price_won"].cpu()

        for i in range(len(batch.titles)):
            if saved >= max_items:
                save_json(out_dir / "attention_explanations.json", explanations)
                return

            field_weights = []
            for j, name in enumerate(batch.field_names[i]):
                field_weights.append(
                    {
                        "field": name,
                        "text": batch.field_texts[i][j],
                        "attention": float(field_attn[i, j].item()),
                    }
                )

            explanations.append(
                {
                    "model_type": "text_only",
                    "item_index": batch.item_indices[i],
                    "title": batch.titles[i],
                    "target_price_won": float(batch.price_won[i].item()),
                    "pred_price_won": float(pred_price[i].item()),
                    "field_weights": sorted(field_weights, key=lambda x: x["attention"], reverse=True),
                    "image_weights": [],
                }
            )
            saved += 1

    save_json(out_dir / "attention_explanations.json", explanations)


def main():
    args = parse_args()
    seed_everything(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, device)
    tokenizer = model.tokenizer

    dataset = TextOnlyPriceRegressionDataset(
        merged_json=args.merged_json,
        segmentation_json=args.segmentation_json,
        max_images_per_item=args.max_images_per_item,
        drop_null_fields=args.drop_null_fields,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training samples found.")

    print(
        f"[Data:text] samples={len(dataset)} "
        f"skipped_no_price={dataset.skipped_no_price} "
        f"skipped_no_segmented_image={dataset.skipped_no_image}"
    )

    val_len = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) > 1 else 0
    train_len = len(dataset) - val_len
    if val_len > 0:
        generator = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator)
    else:
        train_ds, val_ds = dataset, None

    collator = TextOnlyCollator(
        tokenizer=tokenizer,
        max_length=args.max_text_length,
        max_fields=args.max_fields,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
        drop_last=False,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collator,
            drop_last=False,
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_mae = float("inf")
    val_metrics = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args, epoch)
        print(
            f"[Epoch {epoch}] train text "
            f"loss={train_metrics['loss']:.4f} "
            f"MAE={train_metrics['mae_won']:,.0f}won "
            f"RMSE={train_metrics['rmse_won']:,.0f}won "
            f"MAPE={train_metrics['mape_percent']:.2f}%"
        )

        val_metrics = train_metrics
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"[Epoch {epoch}] val   text "
                f"loss={val_metrics['loss']:.4f} "
                f"MAE={val_metrics['mae_won']:,.0f}won "
                f"RMSE={val_metrics['rmse_won']:,.0f}won "
                f"MAPE={val_metrics['mape_percent']:.2f}%"
            )

        if val_metrics["mae_won"] < best_mae:
            best_mae = val_metrics["mae_won"]
            save_checkpoint(ckpt_dir / "price_regression_text_only_best.pt", model, optimizer, epoch, val_metrics, args)

        if epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"price_regression_text_only_epoch{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                val_metrics,
                args,
            )

    save_checkpoint(
        ckpt_dir / "price_regression_text_only_last.pt",
        model,
        optimizer,
        args.epochs,
        val_metrics or {},
        args,
    )

    explain_dir = Path(args.explain_dir) if args.explain_dir else ckpt_dir / "attention_explanations"
    save_explanations(
        model=model,
        loader=val_loader if val_loader is not None else train_loader,
        device=device,
        out_dir=explain_dir,
        max_items=args.explain_count,
    )
    print(f"[Explain:text] saved to {explain_dir}")


if __name__ == "__main__":
    main()
