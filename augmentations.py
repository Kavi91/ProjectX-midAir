from torchvision import transforms
from params import par

def get_transformer(resize_mode: str, new_size: tuple, is_training: bool) -> transforms.Compose:
    """
    Build and return a transformation pipeline.
    
    Args:
        resize_mode (str): Either 'crop' or 'rescale'.
        new_size (tuple): New (height, width) for the image.
        is_training (bool): If True, include data augmentation if enabled.
    
    Returns:
        A torchvision.transforms.Compose object.
    """
    transform_ops = []
    
    # Augmentation for training mode, if enabled
    if is_training and par.enable_augmentation:
        transform_ops.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
    # Resize operation
    if resize_mode == 'crop':
        transform_ops.append(transforms.CenterCrop(new_size))
    elif resize_mode == 'rescale':
        transform_ops.append(transforms.Resize(new_size))
    
    transform_ops.append(transforms.ToTensor())
    return transforms.Compose(transform_ops)