#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from llm_machine.siglip_price_regression import (
    PriceBatch,
    PriceRegressionAttentionModel,
    PriceRegressionCollator,
    PriceRegressionDataset,
    TokenAttentionExtractor,
    regression_metrics_from_log,
    save_json,
)
from llm_machine.text_encoder import LLMTextEncoder
from networks.RetrievalNet_token_multi import Token


PROJECT_DIR = Path("/home/policelab_l40s/llm_prompt/llm_prompt/project")
DEFAULT_MERGED_JSON = PROJECT_DIR / "merged_with_final_description.json"
DEFAULT_SEG_JSON = PROJECT_DIR / "images" / "segmentation_labels.json"
DEFAULT_CKPT_DIR = Path("/home/policelab_l40s/llm_prompt/llm_prompt/llm_machine/checkpoint_price_regression")

NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train price regression with final_description fields and segmented iPhone images."
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
    parser.add_argument("--train-vision", action="store_true")
    parser.add_argument(
        "--freeze-vision-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze token_model.backbone while training the rest of the vision token module.",
    )
    parser.add_argument("--token-ckpt-path", default=None)
    parser.add_argument("--regression-ckpt-path", default=None)
    parser.add_argument("--partial-vt-ckpt-path", default=None)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--explain-count", type=int, default=8)
    parser.add_argument("--explain-dir", default=None)
    return parser.parse_args()


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


def unwrap_state_dict(raw: Any) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "net", "module", "regression_model", "token_model"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    return raw if isinstance(raw, dict) else {}


def strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if sd and all(k.startswith("module.") for k in sd):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def load_matching_weights(model: torch.nn.Module, path: Optional[str], label: str) -> None:
    if not path:
        return
    if not os.path.isfile(path):
        print(f"[Load:{label}] not found: {path}")
        return

    raw = torch.load(path, map_location="cpu", weights_only=False)
    sd = strip_module_prefix(unwrap_state_dict(raw))
    model_sd = model.state_dict()
    matched = {
        k: v
        for k, v in sd.items()
        if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
    }
    missing = model.load_state_dict(matched, strict=False)
    print(f"[Load:{label}] {path}")
    print(f"  matched={len(matched)} missing={len(missing.missing_keys)} unexpected={len(missing.unexpected_keys)}")


def save_checkpoint(
    path: Path,
    model: PriceRegressionAttentionModel,
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
            "regression_model": model.state_dict(),
            "token_model": token_extractor.token_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )
    print(f"[Save] {path}")


def build_transforms(image_size: Tuple[int, int]):
    return T.Compose(
        [
            T.Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )


def build_models(args, device: torch.device):
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

    regression_model = PriceRegressionAttentionModel(
        text_encoder=text_encoder,
        vision_dim=1024,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        huber_beta=args.huber_beta,
    ).to(device)

    token_model = Token(outputdim=1024, classifier_num=3821).to(device)
    token_extractor = TokenAttentionExtractor(token_model).to(device)

    load_matching_weights(token_model, args.token_ckpt_path, "token")
    load_matching_weights(regression_model, args.partial_vt_ckpt_path, "partial_vt_to_regression")

    if args.regression_ckpt_path:
        raw = torch.load(args.regression_ckpt_path, map_location="cpu", weights_only=False)
        if "regression_model" in raw:
            regression_model.load_state_dict(raw["regression_model"], strict=False)
            print(f"[Resume] regression model <- {args.regression_ckpt_path}")
        if "token_model" in raw:
            token_model.load_state_dict(raw["token_model"], strict=False)
            print(f"[Resume] token model <- {args.regression_ckpt_path}")

    if args.train_vision:
        token_extractor.train()
        for name, p in token_extractor.token_model.named_parameters():
            if args.freeze_vision_backbone and name.startswith("backbone."):
                p.requires_grad = False
            else:
                p.requires_grad = True
    else:
        token_extractor.eval()
        for p in token_extractor.parameters():
            p.requires_grad = False

    vision_trainable = sum(p.numel() for p in token_extractor.parameters() if p.requires_grad)
    vision_frozen = sum(p.numel() for p in token_extractor.parameters() if not p.requires_grad)
    print(
        f"[VisionTrain] train_vision={args.train_vision} "
        f"freeze_backbone={args.freeze_vision_backbone} "
        f"trainable={vision_trainable:,} frozen={vision_frozen:,}"
    )

    return regression_model, token_extractor


def extract_image_tokens(
    token_extractor: TokenAttentionExtractor,
    batch: PriceBatch,
    device: torch.device,
    train_vision: bool,
    return_spatial: bool = False,
):
    images = batch.images.to(device, non_blocking=True)
    image_mask = batch.image_mask.to(device, non_blocking=True)
    bsz, num_images, c, h, w = images.shape
    flat_images = images.reshape(bsz * num_images, c, h, w)
    flat_mask = image_mask.reshape(-1)

    valid_images = flat_images[flat_mask]
    context = torch.enable_grad() if train_vision else torch.no_grad()
    with context:
        token_out = token_extractor(valid_images, return_spatial=return_spatial)

    valid_tokens = token_out["tokens"]
    num_tokens = valid_tokens.size(1)
    feat_dim = valid_tokens.size(2)
    flat_tokens = valid_tokens.new_zeros((bsz * num_images, num_tokens, feat_dim))
    flat_tokens[flat_mask] = valid_tokens
    image_tokens = flat_tokens.reshape(bsz, num_images, num_tokens, feat_dim)

    if not return_spatial:
        return image_tokens, {}

    spatial = {}
    for key in ("query_attention", "decoder_attention"):
        if key not in token_out:
            continue
        valid_map = token_out[key]
        map_shape = valid_map.shape[1:]
        flat_map = valid_map.new_zeros((bsz * num_images, *map_shape))
        flat_map[flat_mask] = valid_map
        spatial[key] = flat_map.reshape(bsz, num_images, *map_shape)
    return image_tokens, spatial


def train_one_epoch(
    model: PriceRegressionAttentionModel,
    token_extractor: TokenAttentionExtractor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
    epoch: int,
):
    model.train()
    if args.train_vision:
        token_extractor.train()
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
            field_input_ids=batch.field_input_ids,
            field_attention_mask=batch.field_attention_mask,
            field_mask=batch.field_mask,
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
                f"[Train] epoch={epoch} step={step}/{len(loader)} "
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
    model: PriceRegressionAttentionModel,
    token_extractor: TokenAttentionExtractor,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()
    token_extractor.eval()
    losses = []
    pred_logs = []
    target_logs = []

    for batch in tqdm(loader, desc="Eval", leave=False):
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
    model: PriceRegressionAttentionModel,
    token_extractor: TokenAttentionExtractor,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    max_items: int,
):
    model.eval()
    token_extractor.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    explanations = []
    saved = 0

    try:
        import matplotlib.pyplot as plt
        has_plt = True
    except Exception:
        has_plt = False

    def save_attention_overlay(image_path: str, attention_map: np.ndarray, overlay_path: Path, heatmap_path: Path):
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

        # Segmented PNGs have transparent background. Masking prevents heat from
        # being visually assigned to background pixels.
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

        alpha = Image.fromarray((attn * 150).astype(np.uint8), mode="L").resize(
            base.size,
            Image.Resampling.BILINEAR,
        )
        heat.putalpha(alpha)

        overlay = Image.alpha_composite(base, heat).convert("RGB")
        overlay.save(overlay_path)

        heat_only = Image.fromarray(heat_rgba, mode="RGBA").resize(base.size, Image.Resampling.BILINEAR)
        heat_only.convert("RGB").save(heatmap_path)

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
            field_input_ids=batch.field_input_ids,
            field_attention_mask=batch.field_attention_mask,
            field_mask=batch.field_mask,
            target_log_price=batch.target_log_price,
            return_attention=True,
        )

        image_attn = out["image_attention"].cpu()
        token_attn = out["token_attention"].cpu()
        field_attn = out["field_attention"].cpu()
        pred_price = out["pred_price_won"].cpu()
        gate = out["modality_gate_image"].cpu()

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

            field_weights = []
            for j, name in enumerate(batch.field_names[i]):
                field_weights.append(
                    {
                        "field": name,
                        "text": batch.field_texts[i][j],
                        "attention": float(field_attn[i, j].item()),
                    }
                )

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

                    combined = (
                        token_spatial
                        * token_attn[i, j].view(-1, 1, 1)
                    ).sum(dim=0)
                    combined = combined / combined.max().clamp(min=1e-6)
                    npy_path = item_dir / f"image_{j:02d}_attention_map.npy"
                    np.save(npy_path, combined.numpy())
                    image_info["attention_map_npy"] = str(npy_path)
                    image_info["attention_map_source"] = map_source

                    if has_plt:
                        png_path = item_dir / f"image_{j:02d}_attention_map.png"
                        heatmap_path = item_dir / f"image_{j:02d}_attention_heatmap_only.png"
                        save_attention_overlay(
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
                    "item_index": batch.item_indices[i],
                    "title": batch.titles[i],
                    "target_price_won": float(batch.price_won[i].item()),
                    "pred_price_won": float(pred_price[i].item()),
                    "modality_gate_image": float(gate[i].item()),
                    "field_weights": sorted(field_weights, key=lambda x: x["attention"], reverse=True),
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

    image_size = (args.image_height, args.image_width)
    transform = build_transforms(image_size)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model, token_extractor = build_models(args, device)
    tokenizer = model.tokenizer

    dataset = PriceRegressionDataset(
        merged_json=args.merged_json,
        segmentation_json=args.segmentation_json,
        image_transform=transform,
        max_images_per_item=args.max_images_per_item,
        drop_null_fields=args.drop_null_fields,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "No training samples found. Run Grounded-SAM inference first so segmentation_labels.json has images."
        )

    print(
        f"[Data] samples={len(dataset)} "
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

    collator = PriceRegressionCollator(
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

    param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
    if args.train_vision:
        param_groups.append(
            {"params": [p for p in token_extractor.parameters() if p.requires_grad], "lr": args.vision_lr}
        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, token_extractor, train_loader, optimizer, device, args, epoch)
        print(
            f"[Epoch {epoch}] train "
            f"loss={train_metrics['loss']:.4f} "
            f"MAE={train_metrics['mae_won']:,.0f}won "
            f"RMSE={train_metrics['rmse_won']:,.0f}won "
            f"MAPE={train_metrics['mape_percent']:.2f}%"
        )

        val_metrics = train_metrics
        if val_loader is not None:
            val_metrics = evaluate(model, token_extractor, val_loader, device)
            print(
                f"[Epoch {epoch}] val   "
                f"loss={val_metrics['loss']:.4f} "
                f"MAE={val_metrics['mae_won']:,.0f}won "
                f"RMSE={val_metrics['rmse_won']:,.0f}won "
                f"MAPE={val_metrics['mape_percent']:.2f}%"
            )

        if val_metrics["mae_won"] < best_mae:
            best_mae = val_metrics["mae_won"]
            save_checkpoint(ckpt_dir / "price_regression_best.pt", model, token_extractor, optimizer, epoch, val_metrics, args)

        if epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"price_regression_epoch{epoch:03d}.pt",
                model,
                token_extractor,
                optimizer,
                epoch,
                val_metrics,
                args,
            )

    save_checkpoint(ckpt_dir / "price_regression_last.pt", model, token_extractor, optimizer, args.epochs, val_metrics, args)

    explain_dir = Path(args.explain_dir) if args.explain_dir else ckpt_dir / "attention_explanations"
    explain_loader = val_loader if val_loader is not None else train_loader
    save_explanations(
        model=model,
        token_extractor=token_extractor,
        loader=explain_loader,
        device=device,
        out_dir=explain_dir,
        max_items=args.explain_count,
    )
    print(f"[Explain] saved to {explain_dir}")


if __name__ == "__main__":
    main()
