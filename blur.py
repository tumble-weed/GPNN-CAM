import torch
import torch.nn.functional as F

def gaussian_kernel(window_size, sigma,device='cpu'):
    # Create a 2D Gaussian kernel
    x, y = torch.meshgrid(torch.linspace(-1, 1, window_size), torch.linspace(-1, 1, window_size))
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum() # normalize the kernel
    kernel = kernel.view(1, 1, window_size, window_size) # reshape the kernel to match the convolution shape
    kernel = kernel.to(device)
    return kernel

def blur(input_tensor, window_size=3, sigma=1):
    # Create the Gaussian kernel
    gaussian_kernel_ = gaussian_kernel(window_size, sigma,device=input_tensor.device)
    # Blur the input tensor using the Gaussian kernel
    blurred_tensor = F.conv2d(input_tensor, gaussian_kernel_, padding=(window_size-1)//2, groups=input_tensor.shape[1])
    return blurred_tensor

