#!/usr/bin/env python3
"""Run inference with fine-tuned MedGemma on CTRate dataset."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image
import base64
import io

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "medgemma_27b_it_script"))

from transformers import AutoProcessor, AutoModelForImageTextToText
from medgemma_27b_it_script.scripts.preprocess import CTPreprocessor

DEFAULT_PROMPT = """You are an expert radiologist. Analyze the provided CT slices and generate a comprehensive radiology report.

Your report should follow this structure:
FINDINGS: [Describe the relevant observations in complete sentences]
IMPRESSION: [Summarize the key diagnosis or abnormalities concisely]

Focus only on clinically relevant abnormalities. Be concise and accurate."""


class MedGemmaInference:
    """Inference wrapper for fine-tuned MedGemma."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "bfloat16",
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype

        self._load_model()
        self._load_processor()
        self._load_preprocessor()

    def _load_model(self):
        logger.info(f"Loading model from: {self.model_path}")

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=getattr(torch, self.dtype),
            device_map=self.device,
        )
        self.model.eval()

        if hasattr(self.model, "generation_config"):
            self.model.generation_config.do_sample = False
            self.model.generation_config.pad_token_id = None

        logger.info("Model loaded successfully")

    def _load_processor(self):
        logger.info("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.processor.tokenizer.padding_side = "left"

        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.processor.tokenizer.eos_token_id

        logger.info("Processor loaded successfully")

    def _load_preprocessor(self):
        self.preprocessor = CTPreprocessor(max_slices=40, num_workers=2)

    def preprocess_volume(
        self,
        volume_path: str,
        max_slices: int = 40,
        max_size: int = 512,
    ) -> List[str]:
        logger.info(f"Preprocessing: {volume_path}")
        b64_images = self.preprocessor.low_memory_preprocess(
            volume_path,
            input_format='nifti',
            max_slices=max_slices,
            max_size=max_size,
        )
        logger.info(f"Preprocessed {len(b64_images)} slices")
        return b64_images

    def generate_report(
        self,
        b64_images: List[str],
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 1500,
    ) -> str:
        content = [{"type": "text", "text": prompt}]
        for b64_img in b64_images:
            content.append({"type": "image", "image": f"data:image/jpeg;base64,{b64_img}"})

        messages = [{"role": "user", "content": content}]

        if hasattr(self.model, "hf_device_map"):
            device_map = self.model.hf_device_map
            if device_map:
                first_device = next(iter(device_map.values()))
                if isinstance(first_device, int):
                    device = f"cuda:{first_device}"
                else:
                    device = str(first_device)
            else:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        ).to(device)

        logger.info("Generating report...")
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[-1]
        text = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        return text.strip()

    def __call__(self, volume_path: str, **kwargs) -> str:
        b64_images = self.preprocess_volume(volume_path, **kwargs)
        return self.generate_report(b64_images, **kwargs)


def batch_inference(
    model_path: str,
    data_dir: Path,
    output_dir: Path,
    max_slices: int = 40,
    max_size: int = 512,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = list(data_dir.glob("*.nii.gz")) + list(data_dir.glob("*.nii"))
    logger.info(f"Found {len(nifti_files)} NIfTI files")

    if not nifti_files:
        logger.warning(f"No NIfTI files found in {data_dir}")
        return

    inferencer = MedGemmaInference(model_path)

    success_count = 0
    for i, nifti_path in enumerate(nifti_files, 1):
        logger.info(f"\n[{i}/{len(nifti_files)}] Processing: {nifti_path.name}")

        try:
            report = inferencer(
                str(nifti_path),
                max_slices=max_slices,
                max_size=max_size,
            )

            output_path = output_dir / f"{nifti_path.stem}_report.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

            logger.info(f"Saved: {output_path.name}")
            success_count += 1

        except Exception as e:
            logger.error(f"Failed: {e}")
            continue

    logger.info(f"\n{'='*60}")
    logger.info(f"Batch complete: {success_count}/{len(nifti_files)} successful")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with fine-tuned MedGemma"
    )

    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to fine-tuned model directory")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--volume", type=str,
                       help="Path to single NIfTI volume")
    group.add_argument("--data-dir", type=Path,
                       help="Directory containing NIfTI files for batch processing")

    parser.add_argument("--output", type=str,
                        help="Output file for single volume mode")
    parser.add_argument("--output-dir", type=Path, default="./reports",
                        help="Output directory for batch mode")
    parser.add_argument("--max-slices", type=int, default=40,
                        help="Maximum number of slices per volume")
    parser.add_argument("--max-size", type=int, default=512,
                        help="Maximum image dimension")
    parser.add_argument("--max-new-tokens", type=int, default=1500,
                        help="Maximum tokens to generate")

    args = parser.parse_args()

    if args.volume:
        if not args.output:
            args.output = Path(args.volume).stem + "_report.txt"

        inferencer = MedGemmaInference(args.model_path)
        report = inferencer(
            args.volume,
            max_slices=args.max_slices,
            max_size=args.max_size,
            max_new_tokens=args.max_new_tokens,
        )

        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"Report saved to: {args.output}")

    else:
        batch_inference(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_slices=args.max_slices,
            max_size=args.max_size,
        )


if __name__ == "__main__":
    main()
