import torch
from torch import nn
import numpy as np
import kornia

class Gauss_Pyramid_Conv(nn.Module):
    """
    Code borrowed from: https://github.com/csjliang/LPTN
    """
    def __init__(self, num_high=3, num_blur=4, channels=11):
        super(Gauss_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.num_blur = num_blur
        self.channels = channels
        
    def downsample(self, x):
        return kornia.filters.blur_pool2d(x, kernel_size=3, stride=2)

    def conv_gauss(self, img):
        # Parameters for gaussian_blur2d: (input, kernel_size, sigma)
        return kornia.filters.gaussian_blur2d(img, (3, 3), (1, 1))
    
    def forward(self, img):
        current = img
        pyr = [current]
        for _ in range(self.num_high):
            # Applying gaussian blur 4 times
            for _ in range(self.num_blur):
                current = self.conv_gauss(current)
            # Downsample using blur_pool2d
            down = self.downsample(current)
            current = down
            pyr.append(current)
        return pyr
        
    # def gauss_kernel(self, device=torch.device('cuda'), channels=3):
    #     kernel = torch.tensor([[1., 4., 6., 4., 1],
    #                            [4., 16., 24., 16., 4.],
    #                            [6., 24., 36., 24., 6.],
    #                            [4., 16., 24., 16., 4.],
    #                            [1., 4., 6., 4., 1.]])
    #     kernel /= 256.
    #     kernel = kernel.repeat(channels, 1, 1, 1)
    #     kernel = kernel.to(device)
    #     return kernel

    # def gauss_kernel(self, size=3, sigma=1, channels=11, device=torch.device('cuda')):
    #     """Generate a Gaussian kernel."""
    #     coords = np.linspace(-(size // 2), size // 2, size)
    #     x, y = np.meshgrid(coords, coords)
    #     kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    #     kernel = kernel / kernel.sum()  # Normalize the kernel
    #     kernel = torch.tensor(kernel, dtype=torch.float32).to(device)
    #     kernel = kernel.unsqueeze(0).repeat(channels, 1, 1, 1)
    #     return kernel

    # def downsample(self, x):
    #     return x[:, :, ::2, ::2]

    # def conv_gauss(self, img, kernel):
    #     img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    #     out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    #     return out

    # def forward(self, img):
    #     current = img
    #     pyr = []
    #     for _ in range(self.num_high):
    #         filtered = self.conv_gauss(current, self.kernel)
    #         pyr.append(filtered)
    #         down = self.downsample(filtered)
    #         current = down
    #     pyr.append(current)
    #     return pyr