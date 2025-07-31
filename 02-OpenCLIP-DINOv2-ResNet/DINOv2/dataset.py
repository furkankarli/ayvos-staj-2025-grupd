from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(train_dir, val_dir, test_dir, batch_size=32):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)
    test_ds = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    class_names = train_ds.classes
    return train_loader, val_loader, test_loader, class_names
