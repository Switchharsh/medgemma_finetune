#!/usr/bin/env python3
"""Fine-tune MedGemma on CTRate dataset using QLoRA."""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "medgemma_27b_it_script"))

import torch
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

from medgemma_27b_it_script.scripts.preprocess import (
    CTPreprocessor,
    DEFAULT_MAX_SLICES,
)

DEFAULT_MODEL_ID = "google/medgemma-27b-it"
DEFAULT_PROMPT = """You are an expert radiologist. Analyze the provided CT slices and generate a comprehensive radiology report.

Provide a detailed paragraph describing your observations, followed by a concise summary."""


class CTRateDataset:
    """Dataset loader for CTRate CT volumes and radiology reports."""

    def __init__(
        self,
        data_dir: Path,
        metadata_csv: Path,
        max_slices: int = 40,
        max_size: int = 512,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.metadata_csv = Path(metadata_csv)
        self.max_slices = max_slices
        self.max_size = max_size
        self.max_samples = max_samples

        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("train_split + val_split + test_split must equal 1.0")

        self.metadata = self._load_metadata()
        logger.info(f"Loaded {len(self.metadata)} valid entries from metadata")

        if max_samples is not None and max_samples < len(self.metadata):
            logger.info(f"Limiting dataset to {max_samples} samples")
            volume_ids = list(self.metadata.keys())[:max_samples]
            self.metadata = {k: self.metadata[k] for k in volume_ids}

        self.train_items, self.val_items, self.test_items = self._create_splits(
            train_split, val_split, test_split
        )
        logger.info(f"Train: {len(self.train_items)}, Val: {len(self.val_items)}, Test: {len(self.test_items)}")

        self.preprocessor = None

    def _load_metadata(self) -> Dict[str, Dict[str, str]]:
        metadata = {}

        # Load allowed VolumeNames from stratified sampling
        allowed_volumes = set()
        with open('data/ctrate_train_1k.txt', 'r') as f:
            for line in f:
                allowed_volumes.add(line.strip())

        with open(self.metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                volume_name = row.get('VolumeName', '').strip()
                if not volume_name:
                    continue

                # Check if this volume is in our stratified list
                if volume_name not in allowed_volumes:
                    continue

                volume_path = self.data_dir / volume_name
                if not volume_path.exists():
                    continue

                findings = row.get('Findings_EN', '').strip()
                impressions = row.get('Impressions_EN', '').strip()

                if findings and impressions:
                    report = f"{findings}\n\n{impressions}"
                elif findings:
                    report = findings
                else:
                    continue

                volume_id = volume_name.replace('.nii.gz', '').replace('.nii', '')
                metadata[volume_id] = {
                    'volume_name': volume_name,
                    'volume_path': str(volume_path),
                    'report': report,
                }

        return metadata

    def _create_splits(
        self, train_split: float, val_split: float, test_split: float
    ) -> Tuple[List[str], List[str], List[str]]:
        volume_ids = list(self.metadata.keys())
        n_total = len(volume_ids)

        import random
        random.seed(42)
        random.shuffle(volume_ids)

        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)

        train_ids = volume_ids[:n_train]
        val_ids = volume_ids[n_train:n_train + n_val]
        test_ids = volume_ids[n_train + n_val:]

        return train_ids, val_ids, test_ids

    def get_preprocessor(self):
        if self.preprocessor is None:
            self.preprocessor = CTPreprocessor(
                max_slices=self.max_slices,
                num_workers=2,
            )
        return self.preprocessor

    def preprocess_volume(self, volume_path: str) -> List[str]:
        preprocessor = self.get_preprocessor()
        return preprocessor.low_memory_preprocess(
            volume_path,
            input_format='nifti',
            max_slices=self.max_slices,
            max_size=self.max_size
        )


class CTRateHFDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for CTRate with on-the-fly preprocessing."""

    def __init__(
        self,
        volume_ids: List[str],
        ctrate_dataset: CTRateDataset,
        processor: AutoProcessor,
    ):
        self.volume_ids = volume_ids
        self.ctrate = ctrate_dataset
        self.processor = processor

    def __len__(self):
        return len(self.volume_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        volume_id = self.volume_ids[idx]
        meta = self.ctrate.metadata[volume_id]

        try:
            b64_images = self.ctrate.preprocess_volume(meta['volume_path'])
        except Exception as e:
            logger.error(f"Failed to preprocess {volume_id}: {e}")
            return {
                "volume_id": volume_id,
                "messages": [],
                "images": [],
                "error": str(e),
            }

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DEFAULT_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": meta['report']},
                ],
            },
        ]

        return {
            "volume_id": volume_id,
            "messages": messages,
            "b64_images": b64_images,
        }


def collate_fn(examples: List[Dict[str, Any]], processor: AutoProcessor):
    valid_examples = [ex for ex in examples if "error" not in ex]
    if not valid_examples:
        return None

    texts = []
    all_images = []

    for example in valid_examples:
        messages = example["messages"]
        b64_images = example["b64_images"]

        content = [{"type": "text", "text": DEFAULT_PROMPT}]
        for b64_img in b64_images:
            content.append({"type": "image", "image": f"data:image/jpeg;base64,{b64_img}"})

        messages[0]["content"] = content

        text = processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        ).strip()
        texts.append(text)

        import base64
        import io
        pil_images = []
        for b64_img in b64_images:
            img_data = base64.b64decode(b64_img)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            pil_images.append(img)
        all_images.append(pil_images)

    batch = processor(
        text=texts,
        images=all_images,
        return_tensors="pt",
        padding=True,
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    try:
        boi_token = processor.tokenizer.special_tokens_map.get("boi_token", "<image>")
        eoi_token = processor.tokenizer.special_tokens_map.get("eoi_token", "</image>")
        boi_id = processor.tokenizer.convert_tokens_to_ids(boi_token)
        eoi_id = processor.tokenizer.convert_tokens_to_ids(eoi_token)
        labels[(labels == boi_id) | (labels == eoi_id)] = -100
    except:
        pass

    for img_token_id in [262144, 256002]:
        labels[labels == img_token_id] = -100

    batch["labels"] = labels
    return batch


class SaveMergedModelCallback(TrainerCallback):
    """Save full model with merged LoRA weights after each epoch."""

    def __init__(self, output_dir: Path, processor: AutoProcessor, save_base_only: bool = False):
        self.output_dir = Path(output_dir)
        self.processor = processor
        self.save_base_only = save_base_only

    def on_epoch_end(self, _args, state, _control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0
        model = kwargs.get("model")

        if model is None:
            logger.warning("Model not found in callback kwargs")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"Saving merged model after epoch {epoch + 1}...")

        merged_dir = self.output_dir / f"merged_epoch_{epoch + 1}"
        merged_dir.mkdir(parents=True, exist_ok=True)

        try:
            if hasattr(model, "merge_and_unload"):
                logger.info("Merging LoRA weights into base model...")
                merged_model = model.merge_and_unload()
                logger.info("LoRA weights merged successfully")
            else:
                logger.warning("Model doesn't have merge_and_unload method, saving without merge")
                merged_model = model

            logger.info(f"Saving merged model to: {merged_dir}")
            merged_model.save_pretrained(merged_dir, safe_serialization=True)

            self.processor.save_pretrained(merged_dir)

            epoch_info = {
                "epoch": epoch + 1,
                "global_step": state.global_step,
                "merged_from": str(self.output_dir),
            }
            with open(merged_dir / "epoch_info.json", "w") as f:
                json.dump(epoch_info, f, indent=2)

            logger.info(f"Merged model saved: {merged_dir}")
            logger.info(f"{'='*60}\n")

        except Exception as e:
            logger.error(f"Failed to save merged model: {e}")
            import traceback
            traceback.print_exc()


def setup_model_and_processor(
    model_id: str,
    use_4bit: bool = True,
    dtype: str = "bfloat16",
    offline: bool = False,
) -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        if device_capability[0] < 8:
            logger.warning(f"GPU compute capability {device_capability} may not support bfloat16 well")
    else:
        raise RuntimeError("CUDA GPU required for training")

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, dtype),
            bnb_4bit_quant_storage=getattr(torch, dtype),
        )
        logger.info("Using 4-bit quantization (QLoRA)")

    model_kwargs = {
        "torch_dtype": getattr(torch, dtype),
        "device_map": "auto",
        "quantization_config": bnb_config,
        "offload_buffers": True,
        "local_files_only": offline,
    }

    logger.info(f"Loading model: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        **model_kwargs
    )

    logger.info(f"Loading processor for: {model_id}")
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=offline
    )

    processor.tokenizer.padding_side = "right"
    model.gradient_checkpointing_enable()

    logger.info("Model and processor loaded successfully")
    return model, processor


def setup_lora_config(
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    r: int = 16,
    target_modules: str = "all-linear",
) -> LoraConfig:
    config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=r,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )
    logger.info(f"LoRA config: alpha={lora_alpha}, r={r}, dropout={lora_dropout}")
    return config


def train(
    data_dir: Path,
    metadata_csv: Path,
    output_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    num_train_epochs: int = 3,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_slices: int = 40,
    max_size: int = 512,
    warmup_ratio: float = 0.03,
    max_samples: Optional[int] = None,
    save_merged_model: bool = True,
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    offline: bool = False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MedGemma Fine-tuning on CTRate Dataset")
    logger.info("=" * 60)

    logger.info(f"Loading dataset from {data_dir}")
    ctrate = CTRateDataset(
        data_dir=data_dir,
        metadata_csv=metadata_csv,
        max_slices=max_slices,
        max_size=max_size,
        max_samples=max_samples,
    )

    model, processor = setup_model_and_processor(model_id, offline=offline)
    peft_config = setup_lora_config()

    train_dataset = CTRateHFDataset(ctrate.train_items, ctrate, processor)
    val_dataset = CTRateHFDataset(ctrate.val_items, ctrate, processor)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    def make_collator():
        def collate_fn_wrapper(examples):
            return collate_fn(examples, processor)
        return collate_fn_wrapper

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="steps",
        eval_steps=50,
        learning_rate=learning_rate,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear",
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        report_to="tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
        save_total_limit=2,
    )

    logger.info("Initializing trainer...")

    callbacks = []
    if save_merged_model:
        callbacks.append(SaveMergedModelCallback(output_dir, processor))
        logger.info("Enabled: Save merged model after each epoch")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=make_collator(),
        callbacks=callbacks,
    )

    logger.info("Starting training...")
    logger.info(f"Effective batch size: {per_device_batch_size * gradient_accumulation_steps}")
    trainer.train()

    logger.info("Saving final model...")
    trainer.save_model()
    processor.save_pretrained(output_dir)

    metadata = {
        "model_id": model_id,
        "dataset": {
            "data_dir": str(data_dir),
            "metadata_csv": str(metadata_csv),
            "total_available": len(ctrate.metadata),
            "max_samples_limit": max_samples,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "max_slices": max_slices,
            "max_size": max_size,
        },
        "training": {
            "num_train_epochs": num_train_epochs,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "effective_batch_size": per_device_batch_size * gradient_accumulation_steps,
        },
    }

    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 60)

    return trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune MedGemma on CTRate dataset using QLoRA"
    )

    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing CT volumes (NIfTI files)")
    parser.add_argument("--metadata-csv", type=Path, required=True,
                        help="Path to metadata CSV file with VolumeName, Findings_EN, Impressions_EN")
    parser.add_argument("--output-dir", type=Path, default="./medgemma-ctrate-finetuned",
                        help="Directory to save checkpoints")
    parser.add_argument("--hub-model-id", type=str, default=None,
                        help="HuggingFace Hub model ID")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push model to HuggingFace Hub")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID,
                        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--num-train-epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max-slices", type=int, default=40,
                        help="Maximum number of slices per CT volume")
    parser.add_argument("--max-size", type=int, default=512,
                        help="Maximum image dimension")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples (None = use all)")
    parser.add_argument("--save-merged-model", action="store_true", default=True,
                        help="Save full merged model after each epoch")
    parser.add_argument("--no-save-merged", action="store_true",
                        help="Disable saving merged models")
    parser.add_argument("--offline", action="store_true",
                        help="Use offline mode")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    if not args.metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {args.metadata_csv}")

    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--hub-model-id required when --push-to-hub is set")

    save_merged = args.save_merged_model and not args.no_save_merged

    train(
        data_dir=args.data_dir,
        metadata_csv=args.metadata_csv,
        output_dir=args.output_dir,
        model_id=args.model_id,
        num_train_epochs=args.num_train_epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_slices=args.max_slices,
        max_size=args.max_size,
        max_samples=args.max_samples,
        save_merged_model=save_merged,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
