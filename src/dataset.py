from torchvision import datasets

class CustomDataset(datasets.ImageFolder):
    def __init__(self, root, special_augment_transform=None, general_augment_transform=None, special_classes=None):
        super().__init__(root)
        
        if special_augment_transform is not None and not callable(special_augment_transform):
            raise ValueError("special_augment_transform must be a callable or None")
        if general_augment_transform is not None and not callable(general_augment_transform):
            raise ValueError("general_augment_transform must be a callable or None")

        if special_classes is not None:
            if not isinstance(special_classes, (set, list, tuple)):
                raise TypeError("special_classes must be a set, list, or tuple")
            self.special_classes = set(special_classes)
        else:
            self.special_classes = set()

        self.special_augment_transform = special_augment_transform
        self.general_augment_transform = general_augment_transform

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        class_name = self.classes[label]
        if class_name in self.special_classes and self.special_augment_transform:
            image = self.special_augment_transform(image)
        elif self.general_augment_transform:
            image = self.general_augment_transform(image)

        return image, label
    