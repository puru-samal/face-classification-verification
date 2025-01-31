from torch.utils.data import Dataset
import os
from tqdm import tqdm
from torchvision.io import decode_image

class ImageDataset(Dataset):
    """Custom dataset for loading image-label pairs."""
    def __init__(self, root, transform, num_classes=None, preload=False):
        """
        Args:
            root (str): Path to the directory containing the images folder.
            transform (callable): Transform to be applied to the images.
            num_classes (int, optional): Number of classes to keep. If None, keep all classes.
            preload (bool): If True, preloads all images into memory during initialization.
        """
        self.root = root
        self.labels_file = os.path.join(self.root, "labels.txt")
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = set()
        self.preload = preload
        self.images = []

        # Read image-label pairs from the file
        with open(self.labels_file, 'r') as f:
            lines = f.readlines()

        lines = sorted(lines, key=lambda x: int(x.strip().split(' ')[-1]))

        # Get all unique labels first
        all_labels = sorted(set(int(line.strip().split(' ')[1]) for line in lines))

         # Select subset of classes if specified
        if num_classes is not None:
            selected_classes = set(all_labels[:num_classes])
        else:
            selected_classes = set(all_labels)

        # Store image paths and labels with a progress bar
        for line in tqdm(lines, desc="Loading dataset"):
            img_path, label = line.strip().split(' ')
            label = int(label)
            # Only add if label is in selected classes
            if label in selected_classes:
                self.image_paths.append(os.path.join(self.root, 'images', img_path))
                self.labels.append(label)
                self.classes.add(label)

        assert len(self.image_paths) == len(self.labels), "Images and labels mismatch!"

        # Convert classes to a sorted list
        self.classes = sorted(self.classes)

        # Preload images into memory
        if self.preload:
            self.images = [
                decode_image(img_path, mode='RGB') / 255.0
                for img_path in tqdm(self.image_paths, desc="Preloading images")
            ]

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (transformed image, label)
        """
        # Load and transform image on-the-fly
        #image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.preload:
            image = self.images[idx]
        else:
            image = decode_image(self.image_paths[idx], mode='RGB') / 255.0
        image = self.transform(image)
        label = self.labels[idx]
        return image, label