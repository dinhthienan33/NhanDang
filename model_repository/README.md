# Triton Inference Server Model Repository

This directory is structured to be used with NVIDIA Triton Inference Server.

## Model Structure

The expected format is:

```
model_repository/
├── <model_name>/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.plan
```

Where:
- `<model_name>`: Name of the model (e.g., rain_removal)
- `config.pbtxt`: Model configuration file for Triton
- `1/`: Version directory (you can have multiple versions)
- `model.plan`: The actual TensorRT engine file

## Converting the Model

To convert the PyTorch model to TensorRT format:

1. Run the conversion script:
   ```bash
   python scripts/convert_to_tensorrt.py \
     --model-path model-checkpoint/model_epoch600.pth \
     --output-dir model_repository \
     --model-name rain_removal \
     --precision fp16
   ```

2. This will:
   - Create the appropriate directory structure
   - Generate the config.pbtxt file
   - Convert the PyTorch model to TensorRT format
   - Save the TensorRT engine in the version directory

## TensorRT Optimization

The TensorRT engine is optimized for the specific GPU hardware it was built on. For best performance, build the TensorRT engine on the same hardware that will run the inference server.

## Model Inputs and Outputs

- **Input**: "input" - RGB image tensor of shape [batch_size, 3, 512, 512]
- **Output**: "output" - Processed RGB image tensor of shape [batch_size, 3, 512, 512] 