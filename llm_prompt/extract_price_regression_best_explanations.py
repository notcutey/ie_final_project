#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from llm_machine.siglip_price_regression import PriceRegressionCollator, PriceRegressionDataset
from train_price_regression_attention import (
    DEFAULT_CKPT_DIR,
    DEFAULT_MERGED_JSON,
    DEFAULT_SEG_JSON,
    build_models,
    build_transforms,
    save_explanations,
    seed_everything,
)


DEFAULT_BEST_CKPT = DEFAULT_CKPT_DIR / "price_regression_best.pt"
DEFAULT_EXPLAIN_DIR = DEFAULT_CKPT_DIR / "attention_explanations_best"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained multimodal price regression checkpoint and save attention explanations only."
    )
    parser.add_argument("--merged-json", default=str(DEFAULT_MERGED_JSON))
    parser.add_argument("--segmentation-json", default=str(DEFAULT_SEG_JSON))
    parser.add_argument("--checkpoint", default=str(DEFAULT_BEST_CKPT))
    parser.add_argument("--explain-dir", default=str(DEFAULT_EXPLAIN_DIR))
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
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
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--drop-null-fields", action="store_true")
    parser.add_argument("--explain-count", type=int, default=8)
    parser.add_argument("--train-text-encoder", action="store_true")
    parser.add_argument("--train-vision", action="store_true")
    parser.add_argument("--freeze-vision-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--token-ckpt-path", default=None)
    parser.add_argument("--partial-vt-ckpt-path", default=None)
    parser.add_argument("--regression-ckpt-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.regression_ckpt_path = args.checkpoint
    seed_everything(args.seed)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    transform = build_transforms((args.image_height, args.image_width))
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
        raise RuntimeError("No samples found.")

    val_len = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) > 1 else 0
    train_len = len(dataset) - val_len
    if val_len > 0:
        generator = torch.Generator().manual_seed(args.seed)
        _, explain_ds = random_split(dataset, [train_len, val_len], generator=generator)
    else:
        explain_ds = dataset

    collator = PriceRegressionCollator(
        tokenizer=tokenizer,
        max_length=args.max_text_length,
        max_fields=args.max_fields,
    )
    explain_loader = DataLoader(
        explain_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
        drop_last=False,
    )

    explain_dir = Path(args.explain_dir)
    save_explanations(
        model=model,
        token_extractor=token_extractor,
        loader=explain_loader,
        device=device,
        out_dir=explain_dir,
        max_items=args.explain_count,
    )
    print(f"[Explain] checkpoint={checkpoint}")
    print(f"[Explain] saved to {explain_dir / 'attention_explanations.json'}")


if __name__ == "__main__":
    main()
