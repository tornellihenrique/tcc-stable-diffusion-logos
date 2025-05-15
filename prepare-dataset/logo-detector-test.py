import os
import time
import logging
import shutil

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from icrawler.builtin import GoogleImageCrawler

# ----- CONFIGURATION VARIABLES -----
QUERY = "brand logo"          # Change this query to test different search terms
MAX_IMAGES = 10               # Number of images to download for the query
TEMP_DOWNLOAD_DIR = "./temp_images"      # Temporary folder for downloaded images
RESULTS_DIR = "./detection_results"      # Folder to save annotated images and crops
CONFIDENCE_THRESHOLD = 0.5    # Confidence threshold for detections
MARGIN = 0.05                 # Margin added around detected boxes (fraction of box width/height)
CUSTOM_MODEL_PATH = "./best.pt"  # Path to your custom YOLOv7 model file
# -----------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_images(query: str, max_num: int, download_dir: str):
    """Download images from Google using icrawler."""
    if os.path.exists(download_dir):
        shutil.rmtree(download_dir)
    os.makedirs(download_dir, exist_ok=True)
    logging.info(f"Downloading up to {max_num} images for query: '{query}' into {download_dir}")
    google_crawler = GoogleImageCrawler(storage={'root_dir': download_dir})
    google_crawler.crawl(keyword=query, max_num=max_num)
    logging.info("Download complete.")

def initialize_detector(device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.nn.Module:
    """
    Load your custom YOLOv7 model from a .pt file.
    This uses the official YOLOv7 repo from WongKinYiu via torch.hub.
    """
    logging.info("Loading custom YOLOv7 model for logo detection.")
    try:
        # The 'custom' argument tells torch.hub to load a custom model.
        model = torch.hub.load('WongKinYiu/yolov7', 'custom', CUSTOM_MODEL_PATH, force_reload=True)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Error loading custom YOLOv7 model: {e}")
        raise e

def detect_logos(image: Image.Image, detector: torch.nn.Module, conf_threshold: float, margin: float):
    """
    Run detection on a PIL image using the custom YOLOv7 model.
    Returns:
        boxes: List of bounding boxes [(xmin, ymin, xmax, ymax), ...]
        crops: List of cropped PIL Images corresponding to each box.
    """
    # Convert image to a numpy array (RGB)
    img_np = np.array(image)
    results = detector(img_np)
    # results.xyxy[0] returns detections as (xmin, ymin, xmax, ymax, conf, cls)
    detections = results.xyxy[0].cpu().numpy()
    
    boxes = []
    crops = []
    img_width, img_height = image.size
    
    for det in detections:
        xmin, ymin, xmax, ymax, conf, cls = det
        if conf < conf_threshold:
            continue

        # Clip values and convert to integer pixel coordinates
        xmin = int(max(xmin, 0))
        ymin = int(max(ymin, 0))
        xmax = int(min(xmax, img_width))
        ymax = int(min(ymax, img_height))
        
        # Apply margin.
        x_margin = int((xmax - xmin) * margin)
        y_margin = int((ymax - ymin) * margin)
        xmin = max(0, xmin - x_margin)
        ymin = max(0, ymin - y_margin)
        xmax = min(img_width, xmax + x_margin)
        ymax = min(img_height, ymax + y_margin)
        
        boxes.append((xmin, ymin, xmax, ymax))
        crops.append(image.crop((xmin, ymin, xmax, ymax)))
    return boxes, crops

def annotate_image(image: Image.Image, boxes: list):
    """
    Draw bounding boxes on the image.
    Returns a new annotated image.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        # Attempt to use a truetype font
        font = ImageFont.truetype("arial.ttf", size=16)
    except:
        font = ImageFont.load_default()
    
    for idx, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin-20), f"Logo {idx+1}", fill="red", font=font)
    return annotated

def main():
    # Prepare directories
    if os.path.exists(TEMP_DOWNLOAD_DIR):
        shutil.rmtree(TEMP_DOWNLOAD_DIR)
    os.makedirs(TEMP_DOWNLOAD_DIR, exist_ok=True)
    
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Download images for the query
    download_images(QUERY, MAX_IMAGES, TEMP_DOWNLOAD_DIR)
    
    # Initialize detector
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = initialize_detector(device)
    
    # Process each downloaded image
    for root, _, files in os.walk(TEMP_DOWNLOAD_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.avif')):
                image_path = os.path.join(root, file)
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception as e:
                    logging.error(f"Failed to open image {image_path}: {e}")
                    continue
                
                logging.info(f"Processing image: {image_path}")
                boxes, crops = detect_logos(image, detector, CONFIDENCE_THRESHOLD, MARGIN)
                
                # Save annotated image with drawn bounding boxes
                annotated = annotate_image(image, boxes)
                annotated_path = os.path.join(RESULTS_DIR, f"annotated_{file}")
                annotated.save(annotated_path)
                logging.info(f"Annotated image saved to {annotated_path}")
                
                # Save each crop as a separate image
                for idx, crop in enumerate(crops):
                    crop_path = os.path.join(RESULTS_DIR, f"crop_{idx}_{file}")
                    crop.save(crop_path)
                    logging.info(f"Crop {idx+1} saved to {crop_path}")
    
    logging.info("Detection test complete.")

if __name__ == '__main__':
    main()
