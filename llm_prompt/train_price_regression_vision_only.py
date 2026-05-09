#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from llm_machine.siglip_price_regression import (
    PriceRegressionDataset,
    TokenAttentionExtractor,
    regression_metrics_from_log,
    save_json,
)
from networks.RetrievalNet_token_multi import Token
from train_price_regression_attention import (
    DEFAULT_MERGED_JSON,
    DEFAULT_SEG_JSON,
    build_transforms,
    extract_image_tokens,
    load_matching_weights,
    seed_everything,
)


DEFAULT_CKPT_DIR = Path(
    "/home/policelab_l40s/llm_prompt/llm_prompt/llm_machine/checkpoint_price_regression_vision_only"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a vision-only price regression ablation with segmented iPhone images."
    )
    parser.add_argument("--merged-json", default=str(DEFAULT_MERGED_JSON))
    parser.add_argument("--segmentation-json", default=str(DEFAULT_SEG_JSON))
    parser.add_argument("--ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--vision-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-height", type=int, default=1024)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--max-images-per-item", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--huber-beta", type=float, default=0.15)
    parser.add_argument(
        "--train-vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train Token_Refine/VT attention by default. Use --no-train-vision to freeze all vision modules.",
    )
    parser.add_argument(
        "--freeze-vision-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze token_model.backbone while training the rest of the vision token module.",
    )
    parser.add_argument("--token-ckpt-path", default=None)
    parser.add_argument("--regression-ckpt-path", default=None)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--explain-count", type=int, default=8)
    parser.add_argument("--explain-dir", default=None)
    return parser.parse_args()


@dataclass
class VisionPriceBatch:
    images: torch.Tensor
    image_mask: torch.Tensor
    target_log_price: torch.Tensor
    price_won: torch.Tensor
    field_names: List[List[str]]
    field_texts: List[List[str]]
    image_paths: List[List[str]]
    titles: List[str]
    item_indices: List[int]


class VisionOnlyCollator:
    def __call__(self, batch: Sequence[Dict[str, Any]]) -> VisionPriceBatch:
        batch_size = len(batch)
        max_images = max(len(x["images"]) for x in batch)
        c, h, w = batch[0]["images"][0].shape

        images = torch.zeros(batch_size, max_images, c, h, w, dtype=batch[0]["images"][0].dtype)
        image_mask = torch.zeros(batch_size, max_images, dtype=torch.bool)

        for i, sample in enumerate(batch):
            for j, img in enumerate(sample["images"]):
                images[i, j] = img
                image_mask[i, j] = True

        return VisionPriceBatch(
            images=images,
            image_mask=image_mask,
            target_log_price=torch.tensor([x["target_log_price"] for x in batch], dtype=torch.float32),
            price_won=torch.tensor([x["price_won"] for x in batch], dtype=torch.float32),
            field_names=[x["field_names"] for x in batch],
            field_texts=[x["field_texts"] for x in batch],
            image_paths=[x["image_paths"] for x in batch],
            titles=[x["title"] for x in batch],
            item_indices=[int(x["item_index"]) for x in batch],
        )


class VisionOnlyPriceRegressionModel(nn.Module):
    def __init__(
        self,
        vision_dim: int = 1024,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        huber_beta: float = 0.15,
    ):
        super().__init__()
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

    def aggregate_image_tokens(
        self,
        image_tokens: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_images, num_tokens, _ = image_tokens.shape
        v = self.vision_proj(image_tokens.reshape(bsz * num_images * num_tokens, -1))
        v = v.reshape(bsz, num_images, num_tokens, -1)

        invalid_images = ~image_mask
        token_logits = self.token_score(v).squeeze(-1)
        token_logits = token_logits.masked_fill(invalid_images.unsqueeze(-1), -1e4)
        token_attn = torch.softmax(token_logits, dim=-1)
        image_emb = (v * token_attn.unsqueeze(-1)).sum(dim=2)

        image_logits = self.image_score(image_emb).squeeze(-1)
        image_logits = image_logits.masked_fill(invalid_images, -1e4)
        image_attn = torch.softmax(image_logits, dim=-1)
        visual_emb = (image_emb * image_attn.unsqueeze(-1)).sum(dim=1)
        return visual_emb, image_emb, image_attn, token_attn

    def forward(
        self,
        image_tokens: torch.Tensor,
        image_mask: torch.Tensor,
        target_log_price: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, Any]:
        device = next(self.parameters()).device
        image_tokens = image_tokens.to(device, non_blocking=True)
        image_mask = image_mask.to(device, non_blocking=True)

        visual_emb, image_emb, image_attn, token_attn = self.aggregate_image_tokens(image_tokens, image_mask)
        pred_log_price = self.regressor(visual_emb).squeeze(-1)

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
                    "image_attention": image_attn,
                    "token_attention": token_attn,
                    "image_embeddings": image_emb,
                }
            )

        return out


def build_models(args, device: torch.device):
    model = VisionOnlyPriceRegressionModel(
        vision_dim=1024,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        huber_beta=args.huber_beta,
    ).to(device)

    token_model = Token(outputdim=1024, classifier_num=3821).to(device)
    token_extractor = TokenAttentionExtractor(token_model).to(device)
    load_matching_weights(token_model, args.token_ckpt_path, "token")

    if args.regression_ckpt_path:
        raw = torch.load(args.regression_ckpt_path, map_location="cpu", weights_only=False)
        if "vision_model" in raw:
            model.load_state_dict(raw["vision_model"], strict=False)
            print(f"[Resume] vision model <- {args.regression_ckpt_path}")
        if "token_model" in raw:
            token_model.load_state_dict(raw["token_model"], strict=False)
            print(f"[Resume] token model <- {args.regression_ckpt_path}")

    if args.train_vision:
        set_token_extractor_train_mode(token_extractor, freeze_backbone=args.freeze_vision_backbone)
        for name, p in token_extractor.token_model.named_parameters():
            if name.startswith("backbone."):
                p.requires_grad = not args.freeze_vision_backbone
            elif name.startswith("tr."):
                p.requires_grad = True
            else:
                # ArcFace classifier is not used by TokenAttentionExtractor.forward.
                p.requires_grad = False
    else:
        token_extractor.eval()
        for p in token_extractor.parameters():
            p.requires_grad = False

    vision_trainable = sum(p.numel() for p in token_extractor.parameters() if p.requires_grad)
    vision_frozen = sum(p.numel() for p in token_extractor.parameters() if not p.requires_grad)
    backbone_trainable, backbone_frozen = count_params_by_prefix(token_extractor, "backbone.")
    vt_trainable, vt_frozen = count_params_by_prefix(token_extractor, "tr.")
    classifier_trainable, classifier_frozen = count_params_by_prefix(token_extractor, "classifier.")
    backbone_mode = "train" if token_extractor.token_model.backbone.training else "eval"
    vt_mode = "train" if token_extractor.token_model.tr.training else "eval"
    print("[Backbone] Token.backbone=ResNet(name='resnet101', torchvision.models.resnet101, pretrained=False)")
    print(
        f"[VisionTrain] train_vision={args.train_vision} "
        f"freeze_backbone={args.freeze_vision_backbone} "
        f"trainable={vision_trainable:,} frozen={vision_frozen:,}"
    )
    print(
        f"[VisionTrainDetail] backbone mode={backbone_mode} "
        f"trainable={backbone_trainable:,} frozen={backbone_frozen:,} | "
        f"vt_tr mode={vt_mode} trainable={vt_trainable:,} frozen={vt_frozen:,} | "
        f"classifier trainable={classifier_trainable:,} frozen={classifier_frozen:,}"
    )
    return model, token_extractor


def count_params_by_prefix(token_extractor: TokenAttentionExtractor, prefix: str) -> Tuple[int, int]:
    trainable = 0
    frozen = 0
    for name, p in token_extractor.token_model.named_parameters():
        if not name.startswith(prefix):
            continue
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    return trainable, frozen


def set_token_extractor_train_mode(token_extractor: TokenAttentionExtractor, freeze_backbone: bool) -> None:
    token_extractor.train()
    token_extractor.token_model.tr.train()
    if freeze_backbone:
        token_extractor.token_model.backbone.eval()
    else:
        token_extractor.token_model.backbone.train()


def save_checkpoint(
    path: Path,
    model: VisionOnlyPriceRegressionModel,
    token_extractor: TokenAttentionExtractor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    args: argparse.Namespace,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "vision_model": model.state_dict(),
            "token_model": token_extractor.token_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )
    print(f"[Save] {path}")


def train_one_epoch(
    model: VisionOnlyPriceRegressionModel,
    token_extractor: TokenAttentionExtractor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
    epoch: int,
):
    model.train()
    if args.train_vision:
        set_token_extractor_train_mode(token_extractor, freeze_backbone=args.freeze_vision_backbone)
    else:
        token_extractor.eval()

    losses = []
    pred_logs = []
    target_logs = []
    t0 = time.perf_counter()

    for step, batch in enumerate(loader, start=1):
        image_tokens, _ = extract_image_tokens(
            token_extractor,
            batch,
            device=device,
            train_vision=args.train_vision,
            return_spatial=False,
        )

        optimizer.zero_grad(set_to_none=True)
        out = model(
            image_tokens=image_tokens,
            image_mask=batch.image_mask,
            target_log_price=batch.target_log_price,
            return_attention=False,
        )
        loss = out["loss"]
        loss.backward()
        grad_params = list(model.parameters())
        if args.train_vision:
            grad_params += list(token_extractor.parameters())
        torch.nn.utils.clip_grad_norm_(grad_params, max_norm=1.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        pred_logs.append(out["pred_log_price"].detach().cpu())
        target_logs.append(batch.target_log_price.detach().cpu())

        if step % args.log_every == 0:
            metrics = regression_metrics_from_log(torch.cat(pred_logs), torch.cat(target_logs))
            print(
                f"[Train:vision] epoch={epoch} step={step}/{len(loader)} "
                f"loss={np.mean(losses):.4f} "
                f"MAE={metrics['mae_won']:,.0f}won "
                f"MAPE={metrics['mape_percent']:.2f}% "
                f"time={time.perf_counter() - t0:.1f}s"
            )

    metrics = regression_metrics_from_log(torch.cat(pred_logs), torch.cat(target_logs))
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


@torch.no_grad()
def evaluate(
    model: VisionOnlyPriceRegressionModel,
    token_extractor: TokenAttentionExtractor,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()
    token_extractor.eval()
    losses = []
    pred_logs = []
    target_logs = []

    for batch in tqdm(loader, desc="Eval:vision", leave=False):
        image_tokens, _ = extract_image_tokens(
            token_extractor,
            batch,
            device=device,
            train_vision=False,
            return_spatial=False,
        )
        out = model(
            image_tokens=image_tokens,
            image_mask=batch.image_mask,
            target_log_price=batch.target_log_price,
            return_attention=False,
        )
        losses.append(float(out["loss"].cpu().item()))
        pred_logs.append(out["pred_log_price"].cpu())
        target_logs.append(batch.target_log_price.cpu())

    metrics = regression_metrics_from_log(torch.cat(pred_logs), torch.cat(target_logs))
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def _save_attention_overlay(image_path: str, attention_map: np.ndarray, overlay_path: Path, heatmap_path: Path):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("magma")
    attn = attention_map.astype(np.float32)
    base = Image.open(image_path).convert("RGBA")
    alpha_src = np.asarray(base.getchannel("A")).astype(np.float32) / 255.0
    alpha_small = np.asarray(
        Image.fromarray((alpha_src * 255).astype(np.uint8), mode="L").resize(
            (attn.shape[1], attn.shape[0]),
            Image.Resampling.BILINEAR,
        )
    ).astype(np.float32) / 255.0

    attn = attn * alpha_small
    lo = float(np.percentile(attn[alpha_small > 0.05], 5)) if np.any(alpha_small > 0.05) else float(attn.min())
    hi = float(np.percentile(attn[alpha_small > 0.05], 99)) if np.any(alpha_small > 0.05) else float(attn.max())
    attn = np.clip(attn, lo, hi)
    attn = (attn - lo) / max(hi - lo, 1e-6)
    attn = attn * alpha_small

    white = Image.new("RGBA", base.size, (255, 255, 255, 255))
    base = Image.alpha_composite(white, base)
    heat_rgba = (cmap(attn) * 255).astype(np.uint8)
    heat = Image.fromarray(heat_rgba, mode="RGBA").resize(base.size, Image.Resampling.BILINEAR)
    alpha = Image.fromarray((attn * 150).astype(np.uint8), mode="L").resize(base.size, Image.Resampling.BILINEAR)
    heat.putalpha(alpha)
    Image.alpha_composite(base, heat).convert("RGB").save(overlay_path)
    Image.fromarray(heat_rgba, mode="RGBA").resize(base.size, Image.Resampling.BILINEAR).convert("RGB").save(
        heatmap_path
    )


@torch.no_grad()
def save_explanations(
    model: VisionOnlyPriceRegressionModel,
    token_extractor: TokenAttentionExtractor,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    max_items: int,
):
    model.eval()
    token_extractor.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot  # noqa: F401

        has_plt = True
    except Exception:
        has_plt = False

    explanations = []
    saved = 0

    for batch in loader:
        image_tokens, spatial = extract_image_tokens(
            token_extractor,
            batch,
            device=device,
            train_vision=False,
            return_spatial=True,
        )
        out = model(
            image_tokens=image_tokens,
            image_mask=batch.image_mask,
            target_log_price=batch.target_log_price,
            return_attention=True,
        )

        image_attn = out["image_attention"].cpu()
        token_attn = out["token_attention"].cpu()
        pred_price = out["pred_price_won"].cpu()

        query_maps = spatial.get("query_attention")
        if query_maps is not None:
            query_maps = query_maps.cpu()
        decoder_maps = spatial.get("decoder_attention")
        if decoder_maps is not None:
            decoder_maps = decoder_maps.cpu()

        for i in range(len(batch.titles)):
            if saved >= max_items:
                save_json(out_dir / "attention_explanations.json", explanations)
                return

            item_dir = out_dir / f"item_{batch.item_indices[i]:06d}"
            item_dir.mkdir(parents=True, exist_ok=True)

            image_weights = []
            valid_count = int(batch.image_mask[i].sum().item())
            for j in range(valid_count):
                image_info = {
                    "image_path": batch.image_paths[i][j],
                    "image_attention": float(image_attn[i, j].item()),
                    "token_attention": [float(x) for x in token_attn[i, j].tolist()],
                }

                if decoder_maps is not None or query_maps is not None:
                    if decoder_maps is not None:
                        token_spatial = decoder_maps[i, j].mean(dim=0)
                        map_source = "decoder_attention"
                    else:
                        token_spatial = query_maps[i, j]
                        map_source = "query_attention"

                    combined = (token_spatial * token_attn[i, j].view(-1, 1, 1)).sum(dim=0)
                    combined = combined / combined.max().clamp(min=1e-6)
                    npy_path = item_dir / f"image_{j:02d}_attention_map.npy"
                    np.save(npy_path, combined.numpy())
                    image_info["attention_map_npy"] = str(npy_path)
                    image_info["attention_map_source"] = map_source

                    if has_plt:
                        png_path = item_dir / f"image_{j:02d}_attention_map.png"
                        heatmap_path = item_dir / f"image_{j:02d}_attention_heatmap_only.png"
                        _save_attention_overlay(
                            image_path=batch.image_paths[i][j],
                            attention_map=combined.numpy(),
                            overlay_path=png_path,
                            heatmap_path=heatmap_path,
                        )
                        image_info["attention_map_png"] = str(png_path)
                        image_info["attention_heatmap_only_png"] = str(heatmap_path)

                image_weights.append(image_info)

            explanations.append(
                {
                    "model_type": "vision_only",
                    "item_index": batch.item_indices[i],
                    "title": batch.titles[i],
                    "target_price_won": float(batch.price_won[i].item()),
                    "pred_price_won": float(pred_price[i].item()),
                    "field_weights": [],
                    "image_weights": sorted(image_weights, key=lambda x: x["image_attention"], reverse=True),
                }
            )
            saved += 1

    save_json(out_dir / "attention_explanations.json", explanations)


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transforms((args.image_height, args.image_width))
    model, token_extractor = build_models(args, device)

    dataset = PriceRegressionDataset(
        merged_json=args.merged_json,
        segmentation_json=args.segmentation_json,
        image_transform=transform,
        max_images_per_item=args.max_images_per_item,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training samples found.")

    print(
        f"[Data:vision] samples={len(dataset)} "
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

    collator = VisionOnlyCollator()
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

    param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
    if args.train_vision:
        param_groups.append(
            {"params": [p for p in token_extractor.parameters() if p.requires_grad], "lr": args.vision_lr}
        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    best_mae = float("inf")
    val_metrics = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, token_extractor, train_loader, optimizer, device, args, epoch)
        print(
            f"[Epoch {epoch}] train vision "
            f"loss={train_metrics['loss']:.4f} "
            f"MAE={train_metrics['mae_won']:,.0f}won "
            f"RMSE={train_metrics['rmse_won']:,.0f}won "
            f"MAPE={train_metrics['mape_percent']:.2f}%"
        )

        val_metrics = train_metrics
        if val_loader is not None:
            val_metrics = evaluate(model, token_extractor, val_loader, device)
            print(
                f"[Epoch {epoch}] val   vision "
                f"loss={val_metrics['loss']:.4f} "
                f"MAE={val_metrics['mae_won']:,.0f}won "
                f"RMSE={val_metrics['rmse_won']:,.0f}won "
                f"MAPE={val_metrics['mape_percent']:.2f}%"
            )

        if val_metrics["mae_won"] < best_mae:
            best_mae = val_metrics["mae_won"]
            save_checkpoint(ckpt_dir / "price_regression_vision_only_best.pt", model, token_extractor, optimizer, epoch, val_metrics, args)

        if epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"price_regression_vision_only_epoch{epoch:03d}.pt",
                model,
                token_extractor,
                optimizer,
                epoch,
                val_metrics,
                args,
            )

    save_checkpoint(
        ckpt_dir / "price_regression_vision_only_last.pt",
        model,
        token_extractor,
        optimizer,
        args.epochs,
        val_metrics or {},
        args,
    )

    explain_dir = Path(args.explain_dir) if args.explain_dir else ckpt_dir / "attention_explanations"
    save_explanations(
        model=model,
        token_extractor=token_extractor,
        loader=val_loader if val_loader is not None else train_loader,
        device=device,
        out_dir=explain_dir,
        max_items=args.explain_count,
    )
    print(f"[Explain:vision] saved to {explain_dir}")


if __name__ == "__main__":
    main()
