import sys
sys.path.append("")

import torch
from PIL import Image
from torchvision.transforms import v2


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
    img_path = 'image_test/realtwo.jpg'
    img = Image.open(img_path)
    transform_original = v2.Compose([
        v2.ToTensor(),
        # v2.RandomApply([AddGaussianNoise(0., 0.05)], p=1),
        v2.RandomApply([AddSaltAndPepperNoise(salt_prob=0.02, pepper_prob=0.02)], p=1),
        v2.ToPILImage(),
    ])

    img = transform_original(img)
    img.show()
