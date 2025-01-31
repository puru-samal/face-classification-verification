import torch
import torchvision.transforms.v2 as T


def create_transforms(image_size: int = 112, augment: bool = True) -> T.Compose:
    """Create transform pipeline for face recognition."""
    transform_list = [
        #T.Resize((image_size, image_size)),
        #T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]

    if augment: ## TO DO FOR STUDENTS Default == False
        transform_list.extend([

            # Horizontal flip (common in face recognition)
            T.RandomHorizontalFlip(p=0.5),

            # Very mild geometric transforms
            T.RandomAffine(
                    degrees=5,  # Reduced rotation
                    translate=(0.02, 0.02),  # Reduced translation
                    scale=(0.98, 1.02),  # Reduced scaling
                    interpolation=T.InterpolationMode.BILINEAR
                )
        ])

    # Standard normalization
    transform_list.extend([
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    return T.Compose(transform_list)