from torch.utils.data import Dataset
import os
from tqdm import tqdm
from torchvision.io import decode_image


class ImagePairDataset(Dataset):
    """Custom dataset for loading and transforming image pairs."""
    def __init__(self, root, pairs_file, transform, preload=False):
        """
        Args:
            root (str): Path to the directory containing the images.
            pairs_file (str): Path to the file containing image pairs and match labels.
            transform (callable): Transform to be applied to the images.
        """
        self.root = root
        self.transform = transform

        self.matches = []
        self.image1_paths = []
        self.image2_paths = []
        self.preload = preload
        self.images1 = []
        self.images2 = []

        # Read image pair paths and match labels
        with open(pairs_file, 'r') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Loading dataset"):
            img_path1, img_path2, match = line.strip().split(' ')
            self.image1_paths.append(os.path.join(self.root, img_path1))
            self.image2_paths.append(os.path.join(self.root, img_path2))
            self.matches.append(int(match))  # Convert match to integer

        assert len(self.image1_paths) == len(self.image2_paths) == len(self.matches), "Image pair mismatch"

        # Preload images into memory if specified
        if self.preload:
            self.image1_cache = [
                decode_image(img_path, mode='RGB') / 255.0
                for img_path in tqdm(self.image1_paths, desc="Preloading first images")
            ]
            self.image2_cache = [
                decode_image(img_path, mode='RGB') / 255.0
                for img_path in tqdm(self.image2_paths, desc="Preloading second images")
            ]

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image1_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (transformed image1, transformed image2, match label)
        """
        # Load and transform images on-the-fly
        if self.preload:
            img1 = self.images1[idx]
            img2 = self.images2[idx]
        else:
            img1 = decode_image(self.image1_paths[idx], mode='RGB') / 255.0
            img2 = decode_image(self.image2_paths[idx], mode='RGB') / 255.0
        match = self.matches[idx]

        return self.transform(img1), self.transform(img2), match

