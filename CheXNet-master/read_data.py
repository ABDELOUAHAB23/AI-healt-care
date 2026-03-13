# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir: str, image_list_file: str, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory not found: {data_dir}")
        if not os.path.exists(image_list_file):
            raise ValueError(f"Image list file not found: {image_list_file}")

        image_names = []
        labels = []
        
        try:
            with open(image_list_file, "r") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        items = line.strip().split()
                        if len(items) < 2:
                            logger.warning(f"Skipping invalid line {line_num}: insufficient items")
                            continue
                            
                        image_name = items[0]
                        label = items[1:]
                        
                        try:
                            label = [int(i) for i in label]
                        except ValueError as e:
                            logger.warning(f"Skipping line {line_num}: invalid label format - {e}")
                            continue
                        
                        # Search for the image in subdirectories
                        image_found = False
                        for root, dirs, files in os.walk(data_dir):
                            if image_name in files:
                                image_path = os.path.join(root, image_name)
                                image_found = True
                                image_names.append(image_path)
                                labels.append(label)
                                break
                        
                        if not image_found:
                            logger.warning(f"Image not found in any subdirectory, skipping: {image_name}")
                            
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        continue
                        
        except Exception as e:
            raise RuntimeError(f"Error reading image list file: {e}")

        if not image_names:
            raise ValueError("No valid images found in the dataset")

        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        logger.info(f"Loaded {len(self.image_names)} valid images")

    def __getitem__(self, index: int) -> Tuple[Image.Image, torch.FloatTensor]:
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        try:
            image_name = self.image_names[index]
            with Image.open(image_name) as img:
                image = img.convert('RGB')
                
            label = self.labels[index]
            
            if self.transform is not None:
                try:
                    image = self.transform(image)
                except Exception as e:
                    logger.error(f"Transform error for image {image_name}: {e}")
                    raise
                    
            return image, torch.FloatTensor(label)
            
        except Exception as e:
            logger.error(f"Error loading image at index {index}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.image_names)
