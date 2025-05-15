import os
import argparse
import logging
import time
import shutil
import io
import base64

import pandas as pd
import torch
from PIL import Image

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare dataset for kohya_ss from downloaded parquet files"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the folder containing the 'data' folder with parquet files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output folder to store the prepared dataset (kohya_ss structure)."
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="logo",
        help="Instance prompt."
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="logos",
        help="Class prompt."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        nargs=1,
        default=100,
        help="Repeats."
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default="png",
        help="Image file extension to save (e.g., png or jpg)."
    )
    parser.add_argument(
        "--resize", 
        type=int, 
        nargs=2, 
        default=[512, 512], 
        help="Resize dimensions for images, e.g., --resize 512 512."
    )
    parser.add_argument(
        "--subset_percentage",
        type=float,
        default=100.0,
        help="Percentage of images to include in final dataset (0-100)."
    )
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def load_parquet_files(data_folder: str) -> list:
    """Return a list of .parquet file paths from the given data folder."""
    parquet_files = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if f.endswith('.parquet')
    ]
    if not parquet_files:
        logging.error("No parquet files found in the data folder.")
    else:
        logging.info(f"Found {len(parquet_files)} parquet files.")
    return parquet_files


def load_image_from_data(img_data):
    """
    Load an image from the given data, which can be:
      - bytes: raw image bytes,
      - str: either a file path or a base64-encoded string,
      - dict: expecting a key "bytes" containing either bytes or a base64 string.
    """
    try:
        if isinstance(img_data, bytes):
            return Image.open(io.BytesIO(img_data)).convert("RGB")
        elif isinstance(img_data, str):
            if os.path.exists(img_data):
                return Image.open(img_data).convert("RGB")
            else:
                # Try to decode as base64 string.
                try:
                    decoded = base64.b64decode(img_data)
                    return Image.open(io.BytesIO(decoded)).convert("RGB")
                except Exception as e:
                    logging.error(f"Failed to decode image from string: {e}")
                    return None
        elif isinstance(img_data, dict):
            # Assume the dict contains a key "bytes" with the image content.
            if "bytes" in img_data:
                data_field = img_data["bytes"]
                if isinstance(data_field, bytes):
                    return Image.open(io.BytesIO(data_field)).convert("RGB")
                elif isinstance(data_field, str):
                    try:
                        decoded = base64.b64decode(data_field)
                        return Image.open(io.BytesIO(decoded)).convert("RGB")
                    except Exception as e:
                        logging.error(f"Failed to decode base64 image from dict: {e}")
                        return None
                else:
                    logging.error("The 'bytes' field in image dict is neither bytes nor string.")
                    return None
            else:
                logging.error("Image dict does not contain a 'bytes' field.")
                return None
        else:
            logging.error("Unsupported type for image data.")
            return None
    except Exception as e:
        logging.error(f"Error loading image: {e}")
        return None


def process_dataset(parquet_files: list, output_img_folder: str, image_ext: str, resize_dims: tuple, subset_percentage: float):
    """
    Process each parquet file:
      - Read each row.
      - Optionally sample a subset of rows based on subset_percentage.
      - Convert the image (stored in various forms) into a PIL image.
      - Resize the image.
      - Save the image and a corresponding .txt file containing the caption.
    """
    total_count = 0

    for pf in parquet_files:
        logging.info(f"Processing file: {pf}")
        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            logging.error(f"Failed to read {pf}: {e}")
            continue

        logging.info(f"Loaded {len(df)} samples from {pf}.")
        # Sample a subset if requested.
        if subset_percentage < 100.0:
            df = df.sample(frac=(subset_percentage / 100.0), random_state=42)
            logging.info(f"After sampling, {len(df)} samples will be processed from {pf}.")

        for idx, row in df.iterrows():
            try:
                # Adjust these column names if needed.
                img_data = row["image"]
                caption = row["text"]

                image = load_image_from_data(img_data)
                if image is None:
                    logging.warning(f"Row {idx}: Could not load image. Skipping sample.")
                    continue

                # Resize the image.
                image = image.resize(resize_dims, Image.Resampling.LANCZOS)

                # Create a unique base file name.
                base_name = f"sample_{total_count:06d}"
                image_filename = os.path.join(output_img_folder, f"{base_name}.{image_ext}")
                txt_filename = os.path.join(output_img_folder, f"{base_name}.txt")

                # Save image.
                image.save(image_filename)
                # Save caption.
                with open(txt_filename, "w", encoding="utf-8") as f:
                    f.write(caption)

                total_count += 1
                if total_count % 100 == 0:
                    logging.info(f"Processed {total_count} samples so far...")
            except Exception as e:
                logging.error(f"Error processing row {idx} in file {pf}: {e}")

    logging.info(f"Completed processing. Total samples processed: {total_count}")


def main():
    setup_logging()
    args = parse_args()
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")

    # Expecting that the input directory contains a "data" folder.
    data_folder = os.path.join(args.input_dir, "data")
    if not os.path.exists(data_folder):
        logging.error(f"Data folder not found at: {data_folder}")
        return

    parquet_files = load_parquet_files(data_folder)
    if not parquet_files:
        logging.error("No parquet files found. Exiting.")
        return

    # Create the output structure for kohya_ss.
    os.makedirs(os.path.join(args.output_dir, "img"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "log"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "model"), exist_ok=True)

    images_output_dir = os.path.join(args.output_dir, "img", f"{args.repeats}_{args.instance_prompt} {args.class_prompt}")
    os.makedirs(images_output_dir, exist_ok=True)

    # Process the dataset and convert samples to the required structure.
    process_dataset(parquet_files, images_output_dir, args.image_ext, tuple(args.resize), args.subset_percentage)


if __name__ == '__main__':
    main()
