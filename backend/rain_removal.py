import os
import cv2
import torch
import numpy as np
from typing import Tuple
import math

def numpy2tensor(nump: np.ndarray, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """Convert numpy array to tensor."""
    if nump.dtype == 'uint8':
        nump = (nump/255.)
    nump = (nump - mean) / std
    if nump.ndim == 3:
        nump = nump[np.newaxis,...,]
    tensor = torch.from_numpy(nump.transpose(0,3,1,2).astype('float32'))
    return tensor

def tensor2numpy(tensor: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """Convert tensor to numpy array."""
    arr = tensor.detach().cpu().numpy().transpose(0,2,3,1)
    arr = arr * std + mean 
    arr = np.clip(arr, 0, 1)
    return arr

def expand_size(img: np.ndarray, size: int) -> np.ndarray:
    """Expand image size to be divisible by given size."""
    rows, cols = img.shape[:2]
    nrows, ncols = math.ceil(rows/size) * size, math.ceil(cols/size) * size
    output = np.zeros((nrows, ncols, 3), dtype=img.dtype)
    output[:rows, :cols, :] = img
    return output

def restore_size(img: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Restore image to original size."""
    return img[:,:rows, :cols, :]

class RainRemovalModel:
    def __init__(self, model_path: str):
        """Initialize the rain removal model."""
        from model.find_model import find_model
        
        self.model, _ = find_model('proposed', 'test')
        self.model.load(os.path.dirname(model_path), 
                       epoch=int(model_path.split('epoch')[-1].split('.')[0]))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def process_image(self, img: np.ndarray) -> np.ndarray:
        """Process a single image to remove rain streaks."""
        with torch.no_grad():
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Prepare input
            rows, cols = img_rgb.shape[:2]
            expand_img = expand_size(img_rgb, 256)
            input_tensor = numpy2tensor(expand_img).to(self.device)
            
            # Run inference
            output = self.model.test_one_image(input_tensor)
            
            # Post-process output
            output_img = restore_size(output['output'], rows, cols)
            
            # Convert back to BGR for OpenCV
            output_bgr = cv2.cvtColor(output_img.squeeze() * 255, cv2.COLOR_RGB2BGR)
            
            return output_bgr.astype(np.uint8) 