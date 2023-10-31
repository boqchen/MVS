import importlib
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def divide_into_patches(image):
    assert image.shape == (1, 3, 4096, 4096), f"Expected shape: [1, 3, 4096, 4096], got: {image.shape}"
    patches = [image[:, :, i:i+1024, j:j+1024] for i in range(0, 4096, 1024) for j in range(0, 4096, 1024)]
    return patches

def stitch_patches(patches):
    # Make sure patches is a list of 16 tensors
    assert len(patches) == 16, f"Expected 16 patches, got {len(patches)}"
    
    # Concatenate the patches into 4 rows, each containing 4 patches
    rows = [torch.cat(patches[i:i+4], dim=3) for i in range(0, 16, 4)]
    
    # Concatenate the rows to form the final image
    final_image = torch.cat(rows, dim=2)

    # Ensure the final image is of shape [B, C, 1000, 1000] (B is the batch size)
    assert final_image.shape[2:] == (1024, 1024), f"Expected shape: [B, C, 1024, 1024], got: {final_image.shape}"
    return final_image

def colorize_images(binary_images):
    
    # Create a custom colormap that maps 0 intensity to black, low intensity to blue,
    # medium intensity to green, and high intensity to yellow
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", ["black", "blue", "green", "yellow"])

    rgb_images = []

    for binary_image in binary_images:
        # Remove the channel dimension
        binary_image = binary_image.squeeze(0)

        # Normalize the intensity values to be within [0, 1]
        binary_image = (binary_image - binary_image.min()) / (binary_image.max() - binary_image.min())

        # Ensure the binary image is on the CPU and detached from the computation graph before converting it to a numpy array
        binary_image_np = binary_image.detach().cpu().numpy()

        # Apply the color map
        colored_image = cmap(binary_image_np)

        # Remove the alpha channel
        colored_image_rgb = colored_image[..., :3]

        # Convert the colored image back to a torch tensor
        # And add a dimension for the batch
        colored_image_tensor = torch.from_numpy(colored_image_rgb).float().unsqueeze(0)

        # Append the colored image to the list
        rgb_images.append(colored_image_tensor)

    # Stack the list of tensors into a single tensor
    rgb_images_tensor = torch.cat(rgb_images, dim=0)

    # Permute the dimensions of the tensor to be in the [B, C, H, W] format
    rgb_images_tensor = rgb_images_tensor.permute(0, 3, 1, 2)

    # Now `rgb_images_tensor` is a tensor of shape [B, 3, H, W]
    return rgb_images_tensor


class TBLogger:
    """A utility class for logging data to TensorBoard.

    Attributes:
        log_dir (str): The directory where the TensorBoard logs will be stored.
        tb_logger (SummaryWriter): An instance of the TensorBoard SummaryWriter
            class, which provides methods for writing data to TensorBoard.
    """
    
    def __init__(self, log_dir=None):
        """Initializes the TBLogger class.

        Args:
            log_dir (str, optional): The directory where the TensorBoard logs 
            will be stored. If None, a timestamped subdirectory will be created 
            in the current working directory. Defaults to None.
        """
        self.log_dir = log_dir
        
        # Import the TensorBoard module from PyTorch and create a SummaryWriter instance
        tb_module = importlib.import_module("torch.utils.tensorboard")
        self.tb_logger = getattr(tb_module, "SummaryWriter")(log_dir=self.log_dir)

    def flush(self):
        """Flushes the SummaryWriter instance."""
        self.tb_logger.flush()
        
    def close(self):
        """Closes the SummaryWriter instance, effectively ending the logging session."""
        self.tb_logger.close()

    def run(self, func_name, *args, **kwargs):
        """Runs the specified function of the SummaryWriter instance with the provided arguments.

        Args:
            func_name (str): The name of the function to run.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            The return value of the function that was run, or None if the function does not exist.
        """
        # Check if the function name is "log_scalars"
        if func_name == "log_scalars":
            return self.tb_log_scalars(*args, **kwargs)
        else:
            # Attempt to get the function from the SummaryWriter instance and run it
            tb_log_func = getattr(self.tb_logger, func_name)
            return tb_log_func(*args, **kwargs)

        return None

    def tb_log_scalars(self, metric_dict, step):
        """Logs multiple scalar values to TensorBoard.

        Args:
            metric_dict (dict): A dictionary where the keys are the names of the metrics
                and the values are the corresponding scalar values to be logged.
            step (int): The current step or iteration in the process being logged.
        """
        for k, v in metric_dict.items():
            self.tb_logger.add_scalar(k, v, step)