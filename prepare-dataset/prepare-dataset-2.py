import os
import argparse
import random
import logging

from PIL import Image

import torch
import torch.backends.cudnn as cudnn

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# Import modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Global configurations
dataset_root = '/home/henrique/Projects/tcc/datasets/logo-dataset/'
output_dir = '/home/henrique/Projects/tcc/datasets/logo-dataset-prepared-3/'

instance_prompt = 'logo'
class_prompt = 'logos'
repeats = 40
images_per_category = 100000    # Number of images to select per category
image_size = (512, 512)         # Image resize dimensions

def parse_args():
    parser = argparse.ArgumentParser(description="MiniGPT-4 Dataset Preparation")
    parser.add_argument("--cfg-path", required=True, help="Path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="Specify the GPU to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args

# Parse arguments
args = parse_args()
print('Arguments parsed:', args)

# Load configuration
print('Loading configuration')
cfg = Config(args)

# Initialize the model
print('Initializing model')
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')

# Initialize the chat interface
print('Initializing chat')

conv_dict = {
    'pretrain_vicuna0': CONV_VISION_Vicuna0,
    'pretrain_llama2': CONV_VISION_LLama2
}

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(f'cuda:{args.gpu_id}') for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

chat = Chat(model, vis_processor, device=f'cuda:{args.gpu_id}', stopping_criteria=stopping_criteria)

# Prompt for generating descriptions
prompt = (
    "Analyze the provided logo image and describe its visual elements in detail. "
    "Focus on shapes, colors, text content, and any distinctive features. "
    "If the image doesn't look like a logo, try imagining it as a logo and describe it. "
    "If the image contains multiple logos, create a general description from all of themas they were a single logo and describe it. "
    "Avoid subjective interpretations and provide an objective description suitable for image generation models. "
    "Avoid additional explanations about what you are about to explain, just give the final description."
)

def generate_description(image_path):
    """Generate a detailed description for the given image."""
    try:
        chat_state = CONV_VISION.copy()
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

def process_images():
    """Process images from the dataset root and generate descriptions."""
    try:
        # Initial check for output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        elif any(os.scandir(output_dir)):
            logging.error(f'Non empty output dir: {output_dir}')
            return

        # Setup output_dir
        os.makedirs(os.path.join(output_dir, 'img'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'log'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'model'), exist_ok=True)

        images_output_dir = os.path.join(output_dir, 'img', f'{repeats}_{instance_prompt} {class_prompt}')
        os.makedirs(images_output_dir, exist_ok=True)

        categories = os.listdir(dataset_root)
        for category in categories:
            category_path = os.path.join(dataset_root, category)
            if os.path.isdir(category_path):
                images = []

                for brand in os.listdir(category_path):
                    brand_path = os.path.join(category_path, brand)
                    if os.path.isdir(brand_path):
                        for image in os.listdir(brand_path):
                            name, extension = image.lower().split('.', 1)
                            if extension in ["png", "jpg", "jpeg", "bmp", "tiff", "webp", "avif"]:
                                images.append(os.path.join(brand_path, image))

                # Select random images per category
                selected_images = random.sample(images, min(images_per_category, len(images)))

                for image_index, image_path in enumerate(selected_images):
                    try:
                        # Resize image
                        image = Image.open(image_path).convert('RGB')
                        image = image.resize(image_size, Image.Resampling.LANCZOS)

                        # Define output path
                        image_original_name = os.path.basename(image_path)
                        name, extension = image_original_name.split('.', 1)
                        image_name = f'{name}-{image_index}.{extension}'
                        output_image_path = os.path.join(images_output_dir, image_name)

                        # Save resized image
                        image.save(output_image_path)

                        # Generate and save description
                        description = generate_description(output_image_path)

                        if description:
                            description_path = output_image_path.rsplit('.', 1)[0] + '.txt'

                            with open(description_path, 'w') as f:
                                f.write(description)
                        else:
                            logging.error(f'Empty description for: {image_path}')

                        logging.info(f'Processed image: {image_name}')
                    except Exception as e:
                        logging.error(f'Error processing {image_path}: {e}')

    except Exception as e:
        logging.critical(f'Critical error in image processing: {e}')

if __name__ == '__main__':
    logging.info('Starting logo image processing.')
    process_images()
    logging.info('Finished logo image processing.')
