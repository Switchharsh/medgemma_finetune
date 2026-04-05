# MedGemma Fine-tuning on CTRate

Fine-tune MedGemma-27B on the CTRate dataset using QLoRA.

## Files

- `finetune_medgemma_ctrate.py` - Training script
- `inference_ctrate.py` - Inference script
- `stratified_sampling.py` - Create stratified dataset splits

## Requirements

- GPU with 40GB+ VRAM (A100 recommended)
- Python 3.10+
- See `requirements.txt` for dependencies

## Quick Start

### 1. Prepare dataset

Organize your data as:

```
data_dir/
├── valid_1_a_1.nii.gz
├── valid_1_a_2.nii.gz
└── ...

metadata.csv  # Columns: VolumeName, Findings_EN, Impressions_EN
```

### 2. Fine-tune

```bash
pip install -r requirements.txt

python finetune_medgemma_ctrate.py \
    --data-dir /path/to/CTRATE_Dataset \
    --metadata-csv /path/to/metadata.csv \
    --output-dir ./medgemma-ctrate-finetuned \
    --num-train-epochs 3 \
    --batch-size 2
```

### 3. Run inference

```bash
# Single volume
python inference_ctrate.py \
    --model-path ./medgemma-ctrate-finetuned \
    --volume /path/to/volume.nii.gz \
    --output report.txt

# Batch processing
python inference_ctrate.py \
    --model-path ./medgemma-ctrate-finetuned \
    --data-dir /path/to/CTRATE_Dataset \
    --output-dir ./reports
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 2 | Per-device batch size |
| `--max-slices` | 40 | Max slices per volume |
| `--max-size` | 512 | Max image dimension |
| `--learning-rate` | 2e-4 | Learning rate |
| `--num-train-epochs` | 3 | Training epochs |

## Memory Issues?

Reduce these parameters:
```bash
--batch-size 1 --max-slices 20 --max-size 256
```

## Output

```
medgemma-ctrate-finetuned/
├── adapter_config.json       # LoRA config
├── adapter_model.safetensors # LoRA weights
├── merged_epoch_N/           # Full merged models per epoch
└── training_metadata.json    # Training metadata
```

## License

Apache License 2.0
