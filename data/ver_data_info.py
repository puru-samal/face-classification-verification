import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch


def print_ver_dataset_statistics(val_dataset, test_dataset):
    """
    Print statistics about the verification datasets

    Args:
        val_dataset: Validation ImagePairDataset
        test_dataset: Test ImagePairDataset
    """
    def get_pair_distribution(dataset):
        """Count matching and non-matching pairs"""
        match_counts = {0: 0, 1: 0}  # 0: non-match, 1: match
        for match in dataset.matches:
            match_counts[match] += 1
        return match_counts

    # Get distributions
    val_dist = get_pair_distribution(val_dataset)
    test_dist = get_pair_distribution(test_dataset)

    # Print header
    print("\n" + "="*60)
    print(" "*20 + "Verification Dataset Summary")
    print("="*60)

    # Print overall statistics
    print("\n📊 Overall Statistics:")
    print("-"*60)
    total_pairs = len(val_dataset) + len(test_dataset)
    print(f"Total pairs:          {total_pairs:,}")

    # Function to print split statistics
    def print_split_stats(name, dataset, dist):
        match_ratio = dist[1]/len(dataset)*100
        non_match_ratio = dist[0]/len(dataset)*100

        print(f"\n🔍 {name} Split:")
        print("-"*60)
        print(f"├── Total pairs:      {len(dataset):,}")
        print(f"├── Matching:         {dist[1]:,} pairs ({match_ratio:.1f}%)")
        print(f"└── Non-matching:     {dist[0]:,} pairs ({non_match_ratio:.1f}%)")

    # Print statistics for each split
    print_split_stats("Validation", val_dataset, val_dist)
    print_split_stats("Test", test_dataset, test_dist)

    # Print sample pair information
    print("\n📄 Sample Pair Format:")
    print("-"*60)
    sample_pair = val_dataset[0]
    print(f"├── Image 1:          {sample_pair[0].shape}")
    print(f"├── Image 2:          {sample_pair[1].shape}")
    print(f"└── Match label:      {sample_pair[2]} ({'✓ Match' if sample_pair[2]=='1' else '✗ Non-match'})")

    # Print image properties
    sample_img1, sample_img2, _ = val_dataset[0]
    print("\n🖼️  Image Properties:")
    print("-"*60)
    print(f"├── Shape:            {tuple(sample_img1.shape)} (C, H, W)")
    print(f"├── Value range:      [{sample_img1.min():.1f}, {sample_img1.max():.1f}]")
    print(f"└── Memory size:      {sample_img1.element_size() * sample_img1.nelement() / 1024:.1f} KB/image")

    print("\n" + "="*60 + "\n")



def print_ver_dataloader_info(val_loader, test_loader):
    """Print basic information about the verification dataloaders"""
    # Print header
    print("\n" + "="*60)
    print(" "*20 + "Verification DataLoader Information")
    print("="*60)

    # Calculate total pairs
    total_pairs = (len(val_loader.dataset) + len(test_loader.dataset))

    # Print overall statistics
    print("\n📊 Overall Statistics:")
    print("-"*60)
    print(f"Batch size:           {val_loader.batch_size}")
    print(f"Total pairs:          {total_pairs:,}")
    print(f"Num workers:          {val_loader.num_workers}")
    print(f"Pin memory:           {val_loader.pin_memory}")

    # Print split information
    print("\n📦 Batch Information:")
    print("-"*60)
    print("Validation:")
    print(f"├── Batches:          {len(val_loader):,}")
    print(f"└── Pairs:            {len(val_loader.dataset):,}")

    print("\nTest:")
    print(f"├── Batches:          {len(test_loader):,}")
    print(f"└── Pairs:            {len(test_loader.dataset):,}")

    # Get sample batches
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))

    # Print tensor information
    print("\n📐 Tensor Information:")
    print("-"*70)
    print("Validation:")
    print(f"├── Image 1:          Shape {tuple(val_batch[0].shape)} (N, C, H, W)")
    print(f"├── Image 1 dtype:    {val_batch[0].dtype}")
    print(f"├── Image 2:          Shape {tuple(val_batch[1].shape)} (N, C, H, W)")
    print(f"├── Image 2 dtype:    {val_batch[1].dtype}")
    print(f"├── Labels:           Shape {tuple(val_batch[2].shape)} (N,)")
    print(f"└── Labels dtype:     {val_batch[2].dtype}")

    print("\nTest:")
    print(f"├── Image 1:          Shape {tuple(test_batch[0].shape)} (N, C, H, W)")
    print(f"├── Image 1 dtype:    {test_batch[0].dtype}")
    print(f"├── Image 2:          Shape {tuple(test_batch[1].shape)} (N, C, H, W)")
    print(f"├── Image 2 dtype:    {test_batch[1].dtype}")
    print(f"├── Labels:           Shape {tuple(test_batch[2].shape)} (N,)")
    print(f"└── Labels dtype:     {test_batch[2].dtype}")

    # Calculate and print memory usage
    def get_pair_batch_memory(batch):
        return (batch[0].element_size() * batch[0].nelement() * 2) / (1024 * 1024)  # MB (×2 for pair)

    print("\n💾 Batch Memory Usage:")
    print("-"*60)
    print(f"├── Validation:       {get_pair_batch_memory(val_batch):.1f} MB/batch")
    print(f"└── Test:             {get_pair_batch_memory(test_batch):.1f} MB/batch")

    print("\n" + "="*60 + "\n")



def show_ver_dataset_samples(val_loader, test_loader, samples_per_set=4, figsize=(12, 8)):
    """
    Display verification pairs from validation and test datasets

    Args:
        val_loader: Validation data loader
        test_loader: Test data loader
        samples_per_set: Number of pairs to show from each dataset
        figsize: Figure size (width, height)
    """
    def denormalize(x):
        """Denormalize images from [-1, 1] to [0, 1]"""
        return x * 0.5 + 0.5

    def get_samples(loader, n):
        """Get n samples from a dataloader"""
        batch = next(iter(loader))
        return batch[0][:n], batch[1][:n], batch[2][:n]

    # Get samples from each dataset
    val_imgs1, val_imgs2, val_labels = get_samples(val_loader, samples_per_set)
    test_imgs1, test_imgs2, test_labels = get_samples(test_loader, samples_per_set)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Plot each dataset
    for idx, (imgs1, imgs2, labels, title) in enumerate([
        (val_imgs1, val_imgs2, val_labels, 'Validation Pairs'),
        (test_imgs1, test_imgs2, test_labels, 'Test Pairs')
    ]):
        # Create grids for both images in pairs
        grid1 = make_grid(denormalize(imgs1), nrow=samples_per_set, padding=2)
        grid2 = make_grid(denormalize(imgs2), nrow=samples_per_set, padding=2)

        # Combine grids vertically with space for labels
        combined_grid = torch.cat([grid1, grid2], dim=1)

        # Display grid
        axes[idx].imshow(combined_grid.permute(1, 2, 0).cpu())
        axes[idx].axis('off')
        axes[idx].set_title(title, fontsize=10)

        # Add match/non-match labels with background boxes
        grid_width = grid1.shape[2]
        img_width = grid_width // samples_per_set

        for i, label in enumerate(labels):
            match_text = "✓ Match" if label == 1 else "✗ Non-match"
            color = 'green' if label == 1 else 'red'

            # Add white background box with colored border
            bbox_props = dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec=color,
                alpha=0.8
            )

            axes[idx].text(i * img_width + img_width/2,
                         combined_grid.shape[1] + 15,  # Increased spacing
                         match_text,
                         ha='center',
                         va='top',
                         fontsize=8,
                         color=color,
                         bbox=bbox_props)

    plt.suptitle("Verification Pairs (Top: Image 1, Bottom: Image 2)", y=1.02)
    # Add more bottom margin for labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.show()
