import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def print_cls_dataset_statistics(train_dataset, val_dataset, test_dataset, top_n=10, figsize=(15, 8)):
    """
    Print statistics about the datasets

    Args:
        train_dataset: Training ImageDataset dataset
        val_dataset: Validation ImageDataset dataset
        test_dataset: Test ImageDataset dataset
        top_n: Number of top classes to show in detail
        figsize: Figure size
    """
    def get_class_distribution(dataset):
      """
      Calculate the distribution of classes in the dataset.

      Args:
          dataset (ImageDataset): An instance of the ImageDataset class.

      Returns:
          dict: A dictionary with class names (or labels) as keys and their counts as values.
      """
      class_counts = {}
      for label in dataset.labels:
          class_name = dataset.classes[label]  # Get the class name from the label index
          class_counts[class_name] = class_counts.get(class_name, 0) + 1
      return class_counts

    # Get distributions
    train_dist = get_class_distribution(train_dataset)
    val_dist = get_class_distribution(val_dataset)
    test_dist = get_class_distribution(test_dataset)

    # Sort and get top N classes
    classes = list(train_dist.keys())
    train_counts = np.array([train_dist[c] for c in classes])
    val_counts = np.array([val_dist[c] for c in classes])
    test_counts = np.array([test_dist[c] for c in classes])

    assert len(train_dataset.classes) == len(val_dataset.classes) == len(test_dataset.classes), "Class mismatch!"

    # Print header
    print("\n" + "="*60)
    print(" "*20 + "Classification Dataset Summary")
    print("="*60)

    # Print overall statistics
    print("\n📊 Overall Statistics:")
    print("-"*60)
    print(f"Total classes:        {len(train_dataset.classes):,}")
    print(f"Total images:         {len(train_dataset) + len(val_dataset) + len(test_dataset):,}")
    print("\nSplit sizes:")
    print(f"├── Training:         {len(train_dataset):,} images")
    print(f"├── Validation:       {len(val_dataset):,} images")
    print(f"└── Test:             {len(test_dataset):,} images")

    # Function to print distribution statistics
    def print_dist_stats(name, counts):
        print(f"\n📈 {name} Distribution:")
        print("-"*60)
        print(f"├── Mean:           {counts.mean():.1f} images/class")
        print(f"├── Median:         {np.median(counts):.1f} images/class")
        print(f"├── Min:            {counts.min():.1f} images/class")
        print(f"└── Max:            {counts.max():.1f} images/class")

    # Print distribution statistics for each split
    print_dist_stats("Training Set", train_counts)
    print_dist_stats("Validation Set", val_counts)
    print_dist_stats("Test Set", test_counts)

    # Print sample image properties
    sample_img, _ = train_dataset[0]
    print("\n🖼️  Image Properties:")
    print("-"*60)
    print(f"├── Shape:           {tuple(sample_img.shape)} (C, H, W)")
    print(f"├── Value range:     [{sample_img.min():.1f}, {sample_img.max():.1f}]")
    print(f"└── Memory size:     {sample_img.element_size() * sample_img.nelement() / 1024:.1f} KB/image")

    print("\n" + "="*60 + "\n")



def print_cls_dataloader_info(train_loader, val_loader, test_loader):
    """Print basic information about the dataloaders and batch shapes"""
    # Print header
    print("\n" + "="*60)
    print(" "*20 + "DataLoader Information")
    print("="*60)

    # Calculate total images
    total_images = (len(train_loader.dataset) +
                   len(val_loader.dataset) +
                   len(test_loader.dataset))

    # Print overall statistics
    print("\n📊 Overall Statistics:")
    print("-"*60)
    print(f"Batch size:           {train_loader.batch_size}")
    print(f"Total images:         {total_images:,}")

    # Print split information
    print("\n📦 Batch Information:")
    print("-"*60)
    print("Training:")
    print(f"├── Batches:          {len(train_loader):,}")
    print(f"└── Images:           {len(train_loader.dataset):,}")

    print("\nValidation:")
    print(f"├── Batches:          {len(val_loader):,}")
    print(f"└── Images:           {len(val_loader.dataset):,}")

    print("\nTest:")
    print(f"├── Batches:          {len(test_loader):,}")
    print(f"└── Images:           {len(test_loader.dataset):,}")

    # Get sample batches
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    # Print tensor shapes and dtypes
    print("\n📐 Tensor Information:")
    print("-"*70)
    print("Training:")
    print(f"├── Images:           Shape {tuple(train_batch[0].shape)} (N, C, H, W)")
    print(f"├── Images dtype:     {train_batch[0].dtype}")
    print(f"├── Labels:           Shape {tuple(train_batch[1].shape)} (N,)")
    print(f"└── Labels dtype:     {train_batch[1].dtype}")

    print("\nValidation:")
    print(f"├── Images:           Shape {tuple(val_batch[0].shape)} (N, C, H, W)")
    print(f"├── Images dtype:     {val_batch[0].dtype}")
    print(f"├── Labels:           Shape {tuple(val_batch[1].shape)} (N,)")
    print(f"└── Labels dtype:     {val_batch[1].dtype}")

    print("\nTest:")
    print(f"├── Images:           Shape {tuple(test_batch[0].shape)} (N, C, H, W)")
    print(f"├── Images dtype:     {test_batch[0].dtype}")
    print(f"├── Labels:           Shape {tuple(test_batch[1].shape)} (N,)")
    print(f"└── Labels dtype:     {test_batch[1].dtype}")

    # Calculate and print memory usage
    def get_batch_memory(batch):
        return batch[0].element_size() * batch[0].nelement() / (1024 * 1024)  # MB

    print("\n💾 Batch Memory Usage:")
    print("-"*60)
    print(f"├── Training:         {get_batch_memory(train_batch):.1f} MB/batch")
    print(f"├── Validation:       {get_batch_memory(val_batch):.1f} MB/batch")
    print(f"└── Test:             {get_batch_memory(test_batch):.1f} MB/batch")

    print("\n" + "="*60 + "\n")



def show_cls_dataset_samples(train_loader, val_loader, test_loader, samples_per_set=8, figsize=(10, 6)):
    """
    Display samples from train, validation, and test datasets side by side

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        samples_per_set: Number of samples to show from each dataset
        figsize: Figure size (width, height)
    """
    def denormalize(x):
        """Denormalize images from [-1, 1] to [0, 1]"""
        return x * 0.5 + 0.5

    def get_samples(loader, n):
        """Get n samples from a dataloader"""
        batch = next(iter(loader))
        return batch[0][:n], batch[1][:n]

    # Get samples from each dataset
    train_imgs, train_labels = get_samples(train_loader, samples_per_set)
    val_imgs, val_labels = get_samples(val_loader, samples_per_set)
    test_imgs, test_labels = get_samples(test_loader, samples_per_set)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Plot each dataset
    for idx, (imgs, labels, title) in enumerate([
        (train_imgs, train_labels, 'Training Samples'),
        (val_imgs, val_labels, 'Validation Samples'),
        (test_imgs, test_labels, 'Test Samples')
    ]):

        # Create grid of images
        grid = make_grid(denormalize(imgs), nrow=8, padding=2)

        # Display grid
        axes[idx].imshow(grid.permute(1, 2, 0).cpu())
        axes[idx].axis('off')
        axes[idx].set_title(title, fontsize=10)

        # Add class labels below images (with smaller font)
        grid_width = grid.shape[2]
        imgs_per_row = min(8, samples_per_set)
        img_width = grid_width // imgs_per_row

        for i, label in enumerate(labels):
            col = i % imgs_per_row  # Calculate column position
            class_name = train_loader.dataset.classes[label]
            axes[idx].text(col * img_width + img_width/2,
                         grid.shape[1] + 5,
                         class_name,
                         ha='center',
                         va='top',
                         fontsize=6,
                         rotation=45)

    plt.tight_layout()
    plt.show()


