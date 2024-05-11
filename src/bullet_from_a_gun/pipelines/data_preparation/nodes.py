"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.19.5
"""
import logging
from typing import Dict

from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_images(image_dict: Dict[str, Image.Image]) -> Dict[str, Image.Image]:
    logger.info(image_dict)
    return image_dict

def load_json_annotations(json_dict: Dict[str, str]) -> Dict[str, str]:
    logger.info(json_dict)
    return json_dict
