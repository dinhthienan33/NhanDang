import os 
import cv2  # type: ignore # Ignore cv2 import errors
import argparse
import utils 
import numpy as np 
from model.find_model import find_model
import time
import torch  # type: ignore # Ignore torch import errors
import concurrent.futures
from tqdm import tqdm  # type: ignore # Ignore tqdm import error

class RainRemovalInference:
    def __init__(self, model_name='proposed', ckpt_dir='checkpoint', ckpt_epoch=600, device=None, resize='original', output_size=None):
        self.model_name = model_name
        self.ckpt_dir = ckpt_dir
        self.ckpt_epoch = ckpt_epoch
        self.resize = resize
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.epoch = None
        self.output_size = output_size  # Can be used to upscale output images
        self._load_model()

    def _load_model(self):
        model, _ = find_model(self.model_name, 'test')
        self.epoch = model.load(self.ckpt_dir, epoch=self.ckpt_epoch)
        print(f'Loaded {self.model_name} at EPOCH {self.ckpt_epoch} on {self.device}!!')
        model.generator.eval()
        model.generator_mask.eval()
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
        self.model = model

    def process_image(self, img: np.ndarray, output_size=None) -> np.ndarray:
        """
        Process an image to remove rain
        
        Args:
            img: Input image in BGR format (OpenCV default)
            output_size: Optional tuple (width, height) to resize output
                         None = same as input, 'upscale' = 2x input
        
        Returns:
            Processed image in BGR format
        """
        # Store original size
        original_h, original_w = img.shape[:2]
        
        # Determine output size if not specified
        if output_size is None:
            output_size = self.output_size
            
        if output_size == 'upscale':
            target_w, target_h = original_w * 2, original_h * 2
        elif isinstance(output_size, tuple) and len(output_size) == 2:
            target_w, target_h = output_size
        else:
            target_w, target_h = original_w, original_h
            
        # Convert BGR to RGB (model expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Use 'expand' resize for better quality with arbitrary sizes
        if max(original_h, original_w) > 1024:
            # Resize large images to prevent memory issues
            scale = 1024 / max(original_h, original_w)
            img_rgb = cv2.resize(img_rgb, (int(original_w * scale), int(original_h * scale)))
            
        # Convert to tensor
        input_tensor = utils.numpy2tensor(img_rgb)
        
        # Process with model
        with torch.no_grad():
            output = self.model.test_one_image(input_tensor)
        
        # Get the processed output image
        result_rgb = output['output'].squeeze() * 255
        
        # Resize to target size if different from original
        # if target_w != result_rgb.shape[1] or target_h != result_rgb.shape[0]:
        #     result_rgb = cv2.resize(result_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            
        # Convert back to BGR for OpenCV
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        return result_bgr

    def batch_infer(self, img_dir, save_dir, num_workers=1):
        img_file_list = os.listdir(img_dir)
        img_file_list.sort()
        os.makedirs(save_dir, exist_ok=True)
        def process_image_batch(batch_files):
            results = []
            for filename in batch_files:
                img = cv2.imread(os.path.join(img_dir, filename))
                if img is None:
                    print(f"Warning: Could not read image {filename}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if self.resize == 'square':
                    square_img = cv2.resize(img, (512, 512))
                    input_tensor = utils.numpy2tensor(square_img)
                    with torch.no_grad():
                        output = self.model.test_one_image(input_tensor)
                elif self.resize == 'expand':
                    rows, cols = img.shape[:2]
                    expand_img = utils.expand_size(img, 256)
                    input_tensor = utils.numpy2tensor(expand_img)
                    with torch.no_grad():
                        output = self.model.test_one_image(input_tensor)
                    for title, output_img in output.items():
                        output[title] = utils.restore_size(output_img, rows, cols)
                elif self.resize == 'original':
                    input_tensor = utils.numpy2tensor(img)
                    with torch.no_grad():
                        output = self.model.test_one_image(input_tensor)
                output_path = os.path.join(save_dir, f'{filename[:-4]}.png')
                output_img = cv2.cvtColor(output['output'].squeeze() * 255, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, output_img)
                results.append((filename, output_path))
            return results
        device = self.device
        if device == 'cpu' and num_workers > 1:
            batch_size = max(1, len(img_file_list) // num_workers)
            batches = [img_file_list[i:i + batch_size] for i in range(0, len(img_file_list), batch_size)]
            print(f"Processing {len(img_file_list)} images with {num_workers} workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_image_batch, batch) for batch in batches]
                for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures))):
                    results = future.result()
                    for filename, output_path in results:
                        print(f'Processed: {filename} -> {output_path}')
        else:
            print(f"Processing {len(img_file_list)} images sequentially")
            process_image_batch(tqdm(img_file_list))

# CLI entry point remains the same

def test(args):
    infer = RainRemovalInference(
        model_name=args.model,
        ckpt_dir=args.ckpt_dir,
        ckpt_epoch=args.ckpt_epoch,
        resize=args.resize
    )
    start_time = time.time()
    save_dir = os.path.join(args.save_dir, args.model, str(infer.epoch) + '_' + args.resize)
    infer.batch_infer(args.dataset_dir, save_dir, num_workers=args.num_workers)
    total_time = time.time() - start_time
    print('Test Finished!!')
    print(f'Total processing time: {total_time:.2f} seconds')

if __name__=='__main__':
    parser = argparse.ArgumentParser(prog = 'DeRainDrop')             
    parser.add_argument('--model', default='proposed', type=str, dest='model')
    parser.add_argument('--ckpt_dir', default='checkpoint', type=str, dest='ckpt_dir') 
    parser.add_argument('--ckpt_epoch', default=600, type=int, dest='ckpt_epoch')    
    parser.add_argument('--dataset_dir', default='dataset/', type=str, dest='dataset_dir')    
    parser.add_argument('--save_dir', default='../result/', type=str, dest='save_dir')
    parser.add_argument('--resize', default='original', type=str, dest='resize')
    parser.add_argument('--num_workers', default=4, type=int, dest='num_workers')
    args = parser.parse_args()
    test(args)




