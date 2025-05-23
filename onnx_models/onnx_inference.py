import os
import cv2
import numpy as np
import onnxruntime as ort
import time

class ONNXRainRemovalInference:
    def __init__(self, model_name, ckpt_dir, ckpt_epoch=None, device="CPU", resize="original", output_size=None):
        self.model_name = model_name
        self.ckpt_dir = ckpt_dir
        self.ckpt_epoch = ckpt_epoch
        self.resize = resize
        self.device = device
        self.output_size = output_size
        
        # Set up ONNX Runtime session options
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Set provider based on device
        if self.device.lower() == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print(f"Using CUDA for ONNX inference")
        else:
            self.providers = ["CPUExecutionProvider"]
            print(f"Using CPU for ONNX inference")
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        # Find the mask model file
        mask_model_path = os.path.join(self.ckpt_dir, f"{self.model_name}_mask_epoch{self.ckpt_epoch}.onnx")
        if not os.path.exists(mask_model_path):
            # Try to find any mask model in the directory
            mask_files = [f for f in os.listdir(self.ckpt_dir) if f.startswith(f"{self.model_name}_mask_epoch") and f.endswith(".onnx")]
            if mask_files:
                mask_model_path = os.path.join(self.ckpt_dir, mask_files[0])
                self.ckpt_epoch = mask_files[0].split("_epoch")[1].split(".onnx")[0]
            else:
                raise FileNotFoundError(f"No mask model found in {self.ckpt_dir}")
        
        # Find the generator model file
        generator_model_path = os.path.join(self.ckpt_dir, f"{self.model_name}_generator_epoch{self.ckpt_epoch}.onnx")
        if not os.path.exists(generator_model_path):
            # Try to find any generator model in the directory
            generator_files = [f for f in os.listdir(self.ckpt_dir) if f.startswith(f"{self.model_name}_generator_epoch") and f.endswith(".onnx")]
            if generator_files:
                generator_model_path = os.path.join(self.ckpt_dir, generator_files[0])
                self.ckpt_epoch = generator_files[0].split("_epoch")[1].split(".onnx")[0]
            else:
                raise FileNotFoundError(f"No generator model found in {self.ckpt_dir}")
        
        print(f"Loading mask model from: {mask_model_path}")
        print(f"Loading generator model from: {generator_model_path}")
        
        # Create ONNX Runtime sessions
        self.mask_session = ort.InferenceSession(mask_model_path, self.session_options, providers=self.providers)
        self.generator_session = ort.InferenceSession(generator_model_path, self.session_options, providers=self.providers)
        
        # Get input and output names
        self.mask_input_name = self.mask_session.get_inputs()[0].name
        self.mask_output_name = self.mask_session.get_outputs()[0].name
        
        self.generator_input_name = self.generator_session.get_inputs()[0].name
        self.generator_output_names = [output.name for output in self.generator_session.get_outputs()]
    
    def process_image(self, img, output_size=None):
        """
        Process an image to remove rain
        
        Args:
            img: Input image in BGR format (OpenCV default)
            output_size: Optional tuple (width, height) to resize output
                         None = same as input, "upscale" = 2x input
        
        Returns:
            Processed image in BGR format
        """
        start_time = time.time()
        
        # Store original size
        original_h, original_w = img.shape[:2]
        
        # Determine output size if not specified
        if output_size is None:
            output_size = self.output_size
            
        if output_size == "upscale":
            target_w, target_h = original_w * 2, original_h * 2
        elif isinstance(output_size, tuple) and len(output_size) == 2:
            target_w, target_h = output_size
        else:
            target_w, target_h = original_w, original_h
        
        # Convert BGR to RGB (model expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Use "expand" resize for better quality with arbitrary sizes
        if max(original_h, original_w) > 1024:
            # Resize large images to prevent memory issues
            scale = 1024 / max(original_h, original_w)
            img_rgb = cv2.resize(img_rgb, (int(original_w * scale), int(original_h * scale)))
        
        # Preprocess image for model input
        input_data = self._preprocess_image(img_rgb)
        
        # Run mask model
        mask_output = self.mask_session.run([self.mask_output_name], {self.mask_input_name: input_data})[0]
        
        # Concatenate image and mask for generator input
        combined_input = np.concatenate([input_data, mask_output], axis=1)
        
        # Run generator model
        generator_outputs = self.generator_session.run(
            self.generator_output_names, 
            {self.generator_input_name: combined_input}
        )
        
        # Get the main output (first output is the final result)
        result = generator_outputs[0]
        
        # Postprocess the result
        result_rgb = self._postprocess_image(result)
        
        # Resize to target size if needed
        h, w = result_rgb.shape[:2]
        if target_w != w or target_h != h:
            result_rgb = cv2.resize(result_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert back to BGR for OpenCV
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        print(f"ONNX inference completed in {time.time() - start_time:.2f} seconds")
        
        return result_bgr
    
    def _preprocess_image(self, img):
        """Preprocess image for model input"""
        # Normalize to [0, 1]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        
        # Apply normalization (same as in original model)
        img = (img - 0.5) / 0.5
        
        # Convert to NCHW format
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img.astype(np.float32)
    
    def _postprocess_image(self, img):
        """Postprocess model output to RGB image"""
        # Remove batch dimension and convert to HWC
        img = img.squeeze(0).transpose(1, 2, 0)
        
        # Denormalize
        img = img * 0.5 + 0.5
        
        # Clip to [0, 1]
        img = np.clip(img, 0, 1)
        
        # Convert to uint8
        img = (img * 255).astype(np.uint8)
        
        return img

# Example usage:
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="ONNX Rain Removal Inference")
    parser.add_argument("--model", default="proposed", type=str, help="Model name")
    parser.add_argument("--ckpt_dir", default="onnx_models", type=str, help="Directory containing ONNX models")
    parser.add_argument("--ckpt_epoch", default=None, type=int, help="Model epoch (optional)")
    parser.add_argument("--input", required=True, type=str, help="Input image path")
    parser.add_argument("--output", required=True, type=str, help="Output image path")
    parser.add_argument("--device", default="CPU", type=str, choices=["CPU", "CUDA"], help="Device for inference")
    parser.add_argument("--upscale", action="store_true", help="Upscale output image")
    
    args = parser.parse_args()
    
    # Initialize the model
    model = ONNXRainRemovalInference(
        model_name=args.model,
        ckpt_dir=args.ckpt_dir,
        ckpt_epoch=args.ckpt_epoch,
        device=args.device,
        output_size="upscale" if args.upscale else None
    )
    
    # Load and process the image
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not read image {args.input}")
        sys.exit(1)
    
    # Process the image
    result = model.process_image(img)
    
    # Save the result
    cv2.imwrite(args.output, result)
    print(f"Processed image saved to {args.output}")
