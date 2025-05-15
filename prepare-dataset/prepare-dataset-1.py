import os
import argparse
import random
import logging
import time
from typing import Optional

from PIL import Image

import torch
import torch.backends.cudnn as cudnn

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import (
    Chat,
    CONV_VISION_Vicuna0,
    CONV_VISION_LLama2,
    StoppingCriteriaSub,
)

# Import modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


class ProgressTracker:
    def __init__(self, total: int):
        self.total = total
        self.processed = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.average_time_per_image = 0.0

    def update(self):
        self.processed += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        self.average_time_per_image = elapsed_time / self.processed
        remaining = self.total - self.processed
        eta = self.average_time_per_image * remaining
        percentage = (self.processed / self.total) * 100
        logging.info(
            f'Processed: {self.processed}/{self.total} '
            f'({percentage:.2f}%) | ETA: {self.format_time(eta)}'
        )

    @staticmethod
    def format_time(seconds: float) -> str:
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours}h {mins}m {secs}s"
        elif mins > 0:
            return f"{mins}m {secs}s"
        else:
            return f"{secs}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniGPT-4 Dataset Preparation")
    parser.add_argument("--cfg-path", required=True, help="Path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="Specify the GPU to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help=(
            "Override some settings in the used config. "
            "The key-value pair in xxx=yyy format will be merged into config file "
            "(deprecated, use --cfg-options instead)."
        ),
    )
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    # You can add file handlers or other handlers here if needed
    logging.getLogger().setLevel(logging.INFO)


def load_configuration(args: argparse.Namespace) -> Config:
    logging.info('Loading configuration')
    cfg = Config(args)
    return cfg


def initialize_model(cfg: Config, gpu_id: int) -> torch.nn.Module:
    logging.info('Initializing model')
    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{gpu_id}')
    logging.info('Model initialized successfully')
    return model


def initialize_chat(cfg: Config, model: torch.nn.Module, gpu_id: int) -> Chat:
    logging.info('Initializing chat interface')
    model_config = cfg.model_cfg

    conv_dict = {
        'pretrain_vicuna0': CONV_VISION_Vicuna0,
        'pretrain_llama2': CONV_VISION_LLama2
    }

    conv_key = model_config.model_type
    if conv_key not in conv_dict:
        logging.error(f'Unknown model type: {conv_key}')
        raise ValueError(f'Unknown model type: {conv_key}')
    CONV_VISION = conv_dict[conv_key]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(f'cuda:{gpu_id}') for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device=f'cuda:{gpu_id}', stopping_criteria=stopping_criteria)
    logging.info('Chat interface initialized successfully')
    return chat


def generate_prompt() -> str:
    return (
        "Analyze the provided logo image and describe its visual elements in detail. "
        "Focus on shapes, colors, text content, and any distinctive features. "
        "If the image doesn't look like a logo, try imagining it as a logo and describe it. "
        "If the image contains multiple logos, create a general description from all of them as if they were a single logo and describe it. "
        "Avoid subjective interpretations and provide an objective description suitable for image generation models. "
        "Avoid additional explanations about what you are about to explain, just give the final description."
    )


def generate_description(chat: Chat, image_path: str, conv_vision, prompt: str, gpu_id: int) -> Optional[str]:
    """Generate a detailed description for the given image."""
    try:
        chat_state = conv_vision.copy()
        chat_state.append_message("user", prompt)

        img_list = []
        chat.upload_img(image_path, chat_state, img_list)
        chat.encode_img(img_list)

        # Generate description
        description = chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=1,
            temperature=1.0,
            max_new_tokens=300,
            max_length=2000
        )[0]

        # Reset chat state
        chat_state.messages = []
        img_list = []

        return description
    except Exception as e:
        logging.error(f'Error generating description for {image_path}: {e}')
        return None


def process_images(
    dataset_root: str,
    output_dir: str,
    repeats: int,
    instance_prompt: str,
    class_prompt: str,
    images_per_category: int,
    image_size: tuple,
    chat: Chat,
    cfg: Config
):
    """Process images from the dataset root and generate descriptions."""
    try:
        # Initial check for output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f'Created output directory: {output_dir}')
        elif any(os.scandir(output_dir)):
            logging.error(f'Non-empty output directory: {output_dir}')
            return

        # Setup output directories
        sub_dirs = ['img', 'log', 'model']
        for sub_dir in sub_dirs:
            path = os.path.join(output_dir, sub_dir)
            os.makedirs(path, exist_ok=True)
            logging.debug(f'Ensured directory exists: {path}')

        images_output_dir = os.path.join(output_dir, 'img', f'{repeats}_{instance_prompt} {class_prompt}')
        os.makedirs(images_output_dir, exist_ok=True)
        logging.info(f'Images will be saved to: {images_output_dir}')

        categories = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
        total_images = 0
        category_images = {}

        # Collect images per category
        for category in categories:
            category_path = os.path.join(dataset_root, category)
            images = []

            for brand_or_image in os.listdir(category_path):
                brand_or_image_path = os.path.join(category_path, brand_or_image)

                if os.path.isdir(brand_or_image_path):
                    for image in os.listdir(brand_or_image_path):
                        if not os.path.isfile(os.path.join(brand_or_image_path, image)):
                            continue
                        name, extension = os.path.splitext(image)
                        extension = extension.lower().lstrip('.')
                        if extension in ["png", "jpg", "jpeg", "bmp", "tiff", "webp", "avif"]:
                            images.append(os.path.join(brand_or_image_path, image))
                elif os.path.isfile(brand_or_image_path):
                    name, extension = os.path.splitext(brand_or_image)
                    extension = extension.lower().lstrip('.')
                    if extension in ["png", "jpg", "jpeg", "bmp", "tiff", "webp", "avif"]:
                        images.append(brand_or_image_path)

            selected_images = random.sample(images, min(images_per_category, len(images)))
            category_images[category] = selected_images
            total_images += len(selected_images)
            logging.info(f'Category "{category}": selected {len(selected_images)} images out of {len(images)}')

        progress = ProgressTracker(total=total_images)
        prompt = generate_prompt()

        conv_dict = {
            'pretrain_vicuna0': CONV_VISION_Vicuna0,
            'pretrain_llama2': CONV_VISION_LLama2
        }

        model_config = cfg.model_cfg
        model_type = model_config.model_type
        if model_type not in conv_dict:
            logging.error(f'Unsupported model type for conversation: {model_type}')
            return
        CONV_VISION = conv_dict[model_type]

        for category, images in category_images.items():
            for image_index, image_path in enumerate(images):
                try:
                    # Resize image
                    image = Image.open(image_path).convert('RGB')
                    image = image.resize(image_size, Image.Resampling.LANCZOS)

                    # Define output path
                    image_original_name = os.path.basename(image_path)
                    name, extension = os.path.splitext(image_original_name)
                    image_name = f'{name}-{image_index}{extension}'
                    output_image_path = os.path.join(images_output_dir, image_name)

                    # Save resized image
                    image.save(output_image_path)
                    logging.debug(f'Saved resized image: {output_image_path}')

                    # Generate and save description
                    description = generate_description(chat, output_image_path, CONV_VISION, prompt, chat.device)

                    if description:
                        description_path = os.path.splitext(output_image_path)[0] + '.txt'
                        with open(description_path, 'w', encoding='utf-8') as f:
                            f.write(description)
                        logging.debug(f'Saved description: {description_path}')
                    else:
                        logging.error(f'Empty description for: {image_path}')

                    logging.info(f'Processed image: {image_name}')
                except Exception as e:
                    logging.error(f'Error processing {image_path}: {e}')
                finally:
                    progress.update()

    except Exception as e:
        logging.critical(f'Critical error in image processing: {e}')


def main():
    setup_logging()
    args = parse_args()
    logging.info('Arguments parsed: %s', args)

    cfg = load_configuration(args)
    model = initialize_model(cfg, args.gpu_id)
    chat = initialize_chat(cfg, model, args.gpu_id)

    # Global configurations
    dataset_root = '/home/henrique/Projects/tcc/datasets/logo-dataset/'
    output_dir = '/home/henrique/Projects/tcc/datasets/logo-dataset-prepared-4/'

    instance_prompt = 'logo'
    class_prompt = 'logos'
    repeats = 40
    images_per_category = 500      # Number of images to select per category
    image_size = (512, 512)        # Image resize dimensions

    logging.info('Starting logo image processing.')
    process_images(
        dataset_root=dataset_root,
        output_dir=output_dir,
        repeats=repeats,
        instance_prompt=instance_prompt,
        class_prompt=class_prompt,
        images_per_category=images_per_category,
        image_size=image_size,
        chat=chat,
        cfg=cfg
    )
    logging.info('Finished logo image processing.')


if __name__ == '__main__':
    main()
