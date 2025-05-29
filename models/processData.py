import os
from pathlib import Path
import shutil

from torchvision import datasets, transforms
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, random_split


def extract_data(dir_datazip, dir_data):
    if os.path.exists(dir_data):
        print("Data already extracted.")
        return

    if not os.path.exists(dir_datazip):
        raise FileNotFoundError("Data not found")

    print("Extracting data...")
    shutil.unpack_archive(dir_datazip, dir_data)


def arrange_validation_set(val_images_dir, val_annotations_file, val_processed_dir):
    if not os.path.exists(val_processed_dir):
        print("Processing validation set...")
        os.makedirs(val_processed_dir, exist_ok=True)

        with open(val_annotations_file, "r") as f:
            lines = f.readlines()
        img_to_class = {line.split("\t")[0]: line.split("\t")[1] for line in lines}

        for img_name, class_name in img_to_class.items():
            class_folder = os.path.join(val_processed_dir, class_name)
            os.makedirs(class_folder, exist_ok=True)
            src = os.path.join(val_images_dir, img_name)
            dst = os.path.join(class_folder, img_name)
            if os.path.exists(src):
                shutil.copy(src, dst)


def create_data_loader(dataset, batch_size=32, num_workers=4, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def get_tiny_imagenet_loaders(dir_data_train,
                              dir_data_val_processed,
                              batch_size=32,
                              num_workers=4,
                              img_size=64,
                              val_split=0.2,
                              seed=42,
                              k=5):

    generator = torch.Generator().manual_seed(seed)

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                             std=[0.2302, 0.2265, 0.2262]),
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                             std=[0.2302, 0.2265, 0.2262]),
    ])

    train_dataset = datasets.ImageFolder(dir_data_train, transform=train_transforms)
    full_val_dataset = datasets.ImageFolder(dir_data_val_processed, transform=val_transforms)

    if k > 1:
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        train_loaders = []
        val_loaders = []
        test_loader = create_data_loader(full_val_dataset, batch_size, num_workers, shuffle=False)

        for train_indices, val_indices in kf.split(train_dataset):
            train_subset = torch.utils.data.Subset(train_dataset, train_indices)
            val_subset = torch.utils.data.Subset(train_dataset, val_indices)

            train_loader = create_data_loader(train_subset, batch_size, num_workers, shuffle=True)
            val_loader = create_data_loader(val_subset, batch_size, num_workers, shuffle=False)

            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
        
        return train_loaders, val_loaders, test_loader

    else:
        val_size = int(val_split * len(full_val_dataset))
        test_size = len(full_val_dataset) - val_size
        val_dataset, test_dataset = random_split(full_val_dataset, [val_size, test_size], generator=generator)

        train_loader = create_data_loader(train_dataset, batch_size, num_workers, shuffle=True)
        val_loader = create_data_loader(val_dataset, batch_size, num_workers, shuffle=False)
        test_loader = create_data_loader(test_dataset, batch_size, num_workers, shuffle=False)

        return train_loader, val_loader, test_loader


def prepare_data_and_get_loaders(dir_datazip, dir_data, k=1):
    extract_data(dir_datazip, dir_data)

    dir_data_train = os.path.join(dir_data, "train")
    dir_data_val = os.path.join(dir_data, "val")

    dir_val_images = os.path.join(dir_data_val, "images")
    dir_val_annotations = os.path.join(dir_data_val, "val_annotations.txt")

    dir_data_val_processed = os.path.join(dir_data, "val_processed")
    arrange_validation_set(dir_val_images, dir_val_annotations, dir_data_val_processed)

    return get_tiny_imagenet_loaders(dir_data_train,
                                     dir_data_val_processed,
                                     batch_size=64,
                                     num_workers=8,
                                     img_size=64,
                                     val_split=0.2,
                                     seed=42,
                                     k=k)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = prepare_data_and_get_loaders("data/tiny-imagenet-200.zip")

    assert len(train_loader.dataset) == 100000 , "Expected 100000 training samples"
    assert len(val_loader.dataset) == 2000, "Expected 2000 validation samples"
    assert len(test_loader.dataset) == 8000, "Expected 8000 test samples"

    assert train_loader.batch_size == 64, "Expected batch size of 64"
    assert val_loader.batch_size == 64, "Expected batch size of 64"
    assert test_loader.batch_size == 64, "Expected batch size of 64"

    assert (next(iter(train_loader))[0].shape == (64, 3, 64, 64)), "Expected input shape of (64, 3, 64, 64)"
    assert (next(iter(val_loader))[0].shape == (64, 3, 64, 64)), "Expected input shape of (64, 3, 64, 64)"
    assert (next(iter(test_loader))[0].shape == (64, 3, 64, 64)), "Expected input shape of (64, 3, 64, 64)"

    print("Data loaders are working correctly.")