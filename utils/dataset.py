from typing import Optional, Callable, Union
from torch.utils.data import Dataset
import torch
import os
import glob
import numpy as np
import warnings
import cv2
from PIL import Image

warnings.filterwarnings("ignore")

class BaseDataset(Dataset):
    """Base dataset class with common functionality"""
    def __init__(self, transform: Optional[Callable] = None):
        self.transform = transform
        self.data = []
        self.labels = []
        self._verify_files = True  # Set to False to skip verification for speed
    
    def __getitem__(self, idx: int) -> dict:
        img_path, label = self.data[idx], self.labels[idx]
        
        if isinstance(img_path, str):
            # Try to read the image
            img = cv2.imread(img_path)
            if img is None:
                raise RuntimeError(f"Failed to load image at {img_path}. "
                                 f"Please verify the file exists and is not corrupted.")
            
            img = img[:, :, ::-1]  # BGR to RGB
            img = self.img_normalize(img)
        else:
            img = img_path  # For datasets that already have image data in memory
        
        sample = {'image': img, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def verify_images(self):
        """Verify all images in the dataset can be opened"""
        valid_data = []
        valid_labels = []
        
        print("Verifying dataset images...")
        for idx, (img_path, label) in enumerate(zip(self.data, self.labels)):
            if isinstance(img_path, str):
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        valid_data.append(img_path)
                        valid_labels.append(label)
                    else:
                        print(f"Warning: Failed to load image at {img_path}")
                except Exception as e:
                    print(f"Warning: Error loading image at {img_path}: {str(e)}")
            else:
                valid_data.append(img_path)
                valid_labels.append(label)
                
            if idx % 1000 == 0:
                print(f"Verified {idx}/{len(self.data)} images...")
        
        self.data = valid_data
        self.labels = valid_labels
        print(f"Found {len(self.data)} valid images out of {len(self.data)} total")
    
    def __init_dataset__(self):
        """Initialize dataset and verify images if needed"""
        if self._verify_files:
            self.verify_images()
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def img_normalize(img):
        return img / 255.0

class Caltech256Dataset(BaseDataset):
    def __init__(self, root_dir: str = "caltech256", transform: Optional[Callable] = None):
        super().__init__(transform)
        self.root_dir = os.path.expanduser(root_dir)
        self._classes = 256
        
        for cat in range(self._classes):
            cat_dir = glob.glob(os.path.join(self.root_dir, '%03d*' % (cat + 1)))[0]
            for img_file in glob.glob(os.path.join(cat_dir, '*.jpg')):
                self.data.append(img_file)
                self.labels.append(cat)

class Food101Dataset(BaseDataset):
    def __init__(self, root_dir: str = "food-101", transform: Optional[Callable] = None):
        super().__init__(transform)
        self.root_dir = os.path.expanduser(root_dir)
        self._classes = 101
        
        # Get the base directory (data/food101)
        # base_dir = os.path.dirname(os.path.dirname(root_dir))
        base_dir = os.path.dirname(root_dir)
        
        # Load class mapping from meta directory in the base path
        meta_file = os.path.join(base_dir, 'meta', 'classes.txt')
        with open(meta_file) as f:
            class_names = [line.strip() for line in f.readlines()]
            self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        
        # Load images from the specified split directory (train or test)
        for class_name in class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_file in glob.glob(os.path.join(class_dir, '*.jpg')):
                self.data.append(img_file)
                self.labels.append(self.class_to_idx[class_name])

class CIFAR100Dataset(BaseDataset):
    def __init__(self, root_dir: str = "cifar-100", transform: Optional[Callable] = None):
        super().__init__(transform)
        self.root_dir = os.path.expanduser(root_dir)
        self._classes = 100
        
        # Load data from CIFAR-100 binary files
        import pickle
        
        # Determine if we're loading train or test data
        file_name = 'train' if 'train' in root_dir else 'test'
        
        try:
            with open(os.path.join(root_dir, file_name), 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                
                # CIFAR-100 data is stored in a specific format:
                # - 'data': numpy array of shape (N, 3072) where N is number of images
                # - 'fine_labels': list of class labels
                
                # Reshape data from (N, 3072) to (N, 32, 32, 3)
                # First reshape to (N, 3, 32, 32) then transpose to get correct format
                self.data = entry['data']
                self.data = self.data.reshape(-1, 3, 32, 32)
                self.data = self.data.transpose(0, 2, 3, 1)  # (N, 32, 32, 3)
                
                # Convert to float and normalize to [0, 1]
                self.data = self.data.astype('float32')
                self.data = self.img_normalize(self.data)
                
                self.labels = entry['fine_labels']
                
        except Exception as e:
            raise RuntimeError(f"Error loading CIFAR-100 dataset from {root_dir}: {str(e)}")
    
    def __getitem__(self, idx: int) -> dict:
        img, label = self.data[idx], self.labels[idx]
        
        sample = {'image': img, 'label': label}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class MITIndoor67Dataset(BaseDataset):
    def __init__(self, root_dir: str = "mit67", transform: Optional[Callable] = None):
        super().__init__(transform)
        self.root_dir = os.path.expanduser(root_dir)
        self._classes = 67
        
        # Load class names and create mapping
        # images_dir = os.path.join(root_dir, 'Images')
        images_dir = root_dir
        class_names = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(class_names))}
        
        # breakpoint()
        # Load images
        for class_name in class_names:
            class_dir = os.path.join(images_dir, class_name)
            for img_file in glob.glob(os.path.join(class_dir, '*.jpg')):
                self.data.append(img_file)
                self.labels.append(self.class_to_idx[class_name])
                
        self.__init_dataset__()

# Keep existing transform classes unchanged
class Normalize(object):
    def __init__(self, mean: np.ndarray = np.array([0.485, 0.456, 0.406]),
                 std: np.ndarray = np.array([0.229, 0.224, 0.225])):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = img - self.mean
        img /= self.std
        return {'image': img, 'label': label}

class SquarifyImage(object):
    def __init__(self, box_size: int = 256, scale: tuple = (0.6, 1.2),
                 is_scale: bool = True,
                 seed: Optional[Union[Callable, int]] = None):
        self.box_size = box_size
        self.min_scale_ratio = scale[0]
        self.max_scale_ratio = scale[1]
        self.is_scale = is_scale
        self.seed = seed

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = self.squarify(img)
        return {'image': img, 'label': label}

    def squarify(self, img):
        if self.is_scale:
            img = self.img_scale(img)
        
        w, h, _ = img.shape
        ratio = min(self.box_size / w, self.box_size / h)
        resize_w, resize_h = int(w * ratio), int(h * ratio)
        
        x_pad = (self.box_size - resize_w) // 2
        y_pad = (self.box_size - resize_h) // 2
        
        t_pad, b_pad = x_pad, self.box_size - resize_w - x_pad
        l_pad, r_pad = y_pad, self.box_size - resize_h - y_pad

        resized_img = cv2.resize(img, (resize_h, resize_w))
        img_padded = cv2.copyMakeBorder(resized_img, t_pad, b_pad, l_pad, r_pad,
                                       borderType=0, value=0)
        
        return img_padded

    def img_scale(self, img):
        scale = np.random.uniform(self.min_scale_ratio, self.max_scale_ratio)
        return cv2.resize(img, dsize=None, fx=scale, fy=scale)

class RandomCrop(object):
    def __init__(self, target_size: Union[tuple, int]):
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            assert len(target_size) == 2
            self.target_size = target_size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        h, w = img.shape[:2]
        new_h, new_w = self.target_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[top: top + new_h, left: left + new_w]
        return {'image': img, 'label': label}

class ToTensor(object):
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = img.transpose((2, 0, 1))
        return {
            'image': torch.from_numpy(img),
            'label': torch.tensor(label, dtype=torch.long)
        }