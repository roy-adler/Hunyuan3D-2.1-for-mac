#!/usr/bin/env python3
"""
Simple Hunyuan3D-2.1 Demo for macOS
Generates 3D models from images without the Gradio interface
"""

import sys
import os

# Enable MPS fallback for unsupported operations (must be set before importing torch)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Apply torchvision fixes for MPS compatibility
import torchvision_fix
torchvision_fix.apply_fix()

sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

import torch
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from PIL import Image

print("="*70)
print("Hunyuan3D-2.1 Simple Demo - macOS Edition")
print("="*70)
print()

# Check device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"✅ Using device: MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using device: CUDA")
else:
    device = torch.device("cpu")
    print(f"⚠️  Using device: CPU (no GPU available)")

print(f"PyTorch version: {torch.__version__}")
print()

# Load the model
print("Loading Hunyuan3D shape generation model...")
print("(First run will download ~3-4 GB of model weights)")
print()

# Use float32 for MPS, float16 for CUDA (MPS has better float32 support)
dtype = torch.float32 if device.type == "mps" else torch.float16

try:
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2.1',
        subfolder='hunyuan3d-dit-v2-1',
        device=device,
        dtype=dtype
    )
    
    print(f"✅ Model loaded successfully!")
    print(f"   Device: {pipeline.device}")
    print(f"   Dtype: {pipeline.dtype}")
    
    # Check individual component devices
    if hasattr(pipeline, 'model'):
        model_device = next(pipeline.model.parameters()).device
        print(f"   Model weights on: {model_device}")
    print()
    
    # Generate a mesh
    image_path = 'assets/demo.png'
    output_path = 'output_mesh.glb'
    
    if os.path.exists(image_path):
        print(f"Processing image: {image_path}")
        print(f"Generating 3D mesh on {device}... (this may take 1-2 minutes on MPS)")
        print()
        
        # Generate
        image = Image.open(image_path)
        mesh = pipeline(image=image)[0]
        
        # Save
        mesh.export(output_path)
        print(f"✅ Success! 3D model saved to: {output_path}")
        print()
        print("You can view the .glb file in:")
        print("  - https://gltf-viewer.donmccurdy.com/")
        print("  - Blender")
        print("  - Or drag it into a 3D viewer")
        
    else:
        print(f"❌ Image not found: {image_path}")
        print("Please provide an image file path.")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    import traceback
    traceback.print_exc()
    print()
    print("If you see CUDA errors, the model may have hardcoded CUDA requirements.")
    print("Shape generation should work, but texture generation may need fixes.")

print()
print("="*70)

