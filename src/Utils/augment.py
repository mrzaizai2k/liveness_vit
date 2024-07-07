import sys
sys.path.append("")

import torch
from PIL import Image
from torchvision.transforms import v2
from  torchvision.transforms import InterpolationMode 


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
class AddSaltAndPepperNoise(object):
    def __init__(self, salt_prob=0.5, pepper_prob=0.5):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        
    def __call__(self, tensor):
        assert len(tensor.shape) == 3  # Assume input is a 3D tensor (C, H, W)
        
        c, h, w = tensor.shape
        salt_mask = torch.rand(c, h, w) < self.salt_prob
        pepper_mask = torch.rand(c, h, w) < self.pepper_prob
        
        noisy_tensor = tensor.clone()
        noisy_tensor[salt_mask] = 1.0  # Assuming the range of tensor is [0, 1]
        noisy_tensor[pepper_mask] = 0.0  # Assuming the range of tensor is [0, 1]
        
        return noisy_tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(salt_prob={0}, pepper_prob={1})'.format(self.salt_prob, self.pepper_prob)


if __name__ == "__main__":
    img_path = 'image_test/fakeone.jpg'
    img = Image.open(img_path)
    IMG_SIZE=224
    transform_original = v2.Compose([
        # v2.TrivialAugmentWide(),
        # v2.RandomRotation(degrees= 20),
        v2.Resize(232, interpolation=InterpolationMode.BICUBIC,),
        v2.CenterCrop(IMG_SIZE),
        # v2.RandomApply([v2.RandomResizedCrop(IMG_SIZE)] , p=0.7),
        # v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        # v2.RandomAdjustSharpness(sharpness_factor = 2,p=0.5),
        # v2.RandomAdjustSharpness(sharpness_factor = 0, p=0.5),
        # v2.RandomHorizontalFlip(p=0.3),
        # v2.RandomVerticalFlip(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))] , p=1),
        # v2.RandomGrayscale(p=0.1),
        # v2.JPEG((5, 50)),
        v2.ToTensor(),
        # v2.RandomApply([AddSaltAndPepperNoise(salt_prob=0.02, pepper_prob=0.02)], p=0.5),
        # v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # v2.RandomErasing(p=0.1),
        v2.ToPILImage(),
    ])

    img = transform_original(img)
    img.show()
