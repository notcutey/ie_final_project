# Price Regression Code Export

This folder contains the price-regression related code extracted from:

`/home/policelab_l40s/llm_prompt/llm_prompt`

Included code:

- `train_price_regression_attention.py`
- `train_price_regression_vision_only.py`
- `train_price_regression_text_only.py`
- `extract_price_regression_best_explanations.py`
- `llm_machine/siglip_price_regression.py`
- `llm_machine/text_encoder.py`
- `networks/RetrievalNet_token_multi.py`
- `networks/backbone.py`
- `utils/*.py` needed by the ResNet backbone import path

Not copied:

- checkpoint files such as `*.pt`
- segmented image files
- full Grounded-SAM project code
- `__pycache__` files

The copied scripts still use the original absolute default paths for data and checkpoints, for example:

- `./llm_prompt/llm_prompt/project/merged_with_final_description.json`
- `./llm_prompt/llm_prompt/project/images/segmentation_labels.json`
- `./llm_prompt/llm_prompt/llm_machine/checkpoint_price_regression`

Run from this folder:

```bash
cd ./price_regression_code/llm_prompt

python train_price_regression_attention.py
python train_price_regression_vision_only.py
python train_price_regression_text_only.py
python extract_price_regression_best_explanations.py
```
