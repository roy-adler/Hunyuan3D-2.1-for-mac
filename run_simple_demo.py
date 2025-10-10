#!/usr/bin/env python3
"""
Simple Hunyuan3D-2.1 Demo for macOS
Generates 3D models from images without the Gradio interface
"""

import sys
import os

# ============================================================================
# PERFORMANCE SETTINGS - Adjust these for speed vs quality tradeoff
# ============================================================================
# Preset options: 'test', 'minimum', 'fast', 'balanced', 'quality'
QUALITY_PRESET = 'test'  # Change this to 'minimum', 'fast', 'balanced', or 'quality' for better results

# Manual settings (uncomment to override preset):
# num_inference_steps = 20      # Test: 12 (risky!), Min: 15, Fast: 20, Balanced: 35, Quality: 50+
# octree_resolution = 256       # Test: 96, Min: 192, Fast: 256, Balanced: 320, Quality: 384
# guidance_scale = 3.0          # Test: 1.5, Min: 2.0, Fast: 3.0, Balanced: 4.0, Quality: 5.0
# num_chunks = 4000             # Test: 1500, Min: 3000, Fast: 4000, Balanced: 6000, Quality: 8000
#
# WARNING: 'test' preset is experimental - may fail or produce garbage!
# ============================================================================

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

# Define quality presets
PRESETS = {
    'test': {
        'num_inference_steps': 12,      # RISKY - might fail, ultra fast testing
        'octree_resolution': 96,        # Very blocky/low-poly
        'guidance_scale': 1.5,          # Loose interpretation
        'num_chunks': 1500,             # Minimal chunks (might fail)
    },
    'minimum': {
        'num_inference_steps': 15,      # Minimum steps
        'octree_resolution': 192,       # Lower resolution mesh
        'guidance_scale': 2.0,          # Less guidance = faster
        'num_chunks': 3000,             # Fewer chunks = less memory
    },
    'fast': {
        'num_inference_steps': 20,      # Much faster, decent quality
        'octree_resolution': 256,       # Lower resolution mesh
        'guidance_scale': 3.0,          # Less guidance = faster
        'num_chunks': 4000,             # Fewer chunks = less memory
    },
    'balanced': {
        'num_inference_steps': 35,      # Good balance
        'octree_resolution': 320,       # Medium resolution
        'guidance_scale': 4.0,          # Moderate guidance
        'num_chunks': 6000,             # Moderate chunks
    },
    'quality': {
        'num_inference_steps': 50,      # Default quality
        'octree_resolution': 384,       # High resolution
        'guidance_scale': 5.0,          # Default guidance
        'num_chunks': 8000,             # Default chunks
    }
}

# Apply preset or use custom settings
if QUALITY_PRESET in PRESETS:
    settings = PRESETS[QUALITY_PRESET]
else:
    print(f"Warning: Unknown preset '{QUALITY_PRESET}', using 'balanced'")
    settings = PRESETS['balanced']

# Override with manual settings if defined
if 'num_inference_steps' in dir():
    settings['num_inference_steps'] = num_inference_steps
if 'octree_resolution' in dir():
    settings['octree_resolution'] = octree_resolution
if 'guidance_scale' in dir():
    settings['guidance_scale'] = guidance_scale
if 'num_chunks' in dir():
    settings['num_chunks'] = num_chunks

# Validate settings
MIN_STEPS = 10  # Allow test preset with 12 steps
if settings['num_inference_steps'] < MIN_STEPS:
    print(f"❌ ERROR: num_inference_steps ({settings['num_inference_steps']}) is too low!")
    print(f"   Minimum: {MIN_STEPS} steps")
    print(f"   The model will likely fail to generate a valid mesh.")
    print()
    import sys
    sys.exit(1)
elif settings['num_inference_steps'] < 15 and QUALITY_PRESET != 'test':
    print(f"⚠️  WARNING: Using {settings['num_inference_steps']} steps is risky!")
    print(f"   Results may be poor quality or fail completely.")
    print()

print("="*70)
print("Hunyuan3D-2.1 Simple Demo - macOS Edition")
print("="*70)
print(f"Quality Preset: {QUALITY_PRESET.upper()}")
print(f"  Inference Steps: {settings['num_inference_steps']}")
print(f"  Octree Resolution: {settings['octree_resolution']}")
print(f"  Guidance Scale: {settings['guidance_scale']}")
print(f"  Memory Chunks: {settings['num_chunks']}")
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
        
        # Estimate time based on preset
        time_estimates = {
            'test': '5-7 minutes (EXPERIMENTAL - may fail!)',
            'minimum': '7-9 minutes',
            'fast': '9-11 minutes',
            'balanced': '13-16 minutes',
            'quality': '18-22 minutes'
        }
        est_time = time_estimates.get(QUALITY_PRESET, '10-15 minutes')
        print(f"Generating 3D mesh on {device}... (estimated: {est_time})")
        print()
        
        # Generate with custom settings
        image = Image.open(image_path)
        result = pipeline(
            image=image,
            num_inference_steps=settings['num_inference_steps'],
            octree_resolution=settings['octree_resolution'],
            guidance_scale=settings['guidance_scale'],
            num_chunks=settings['num_chunks']
        )
        
        # Check if generation was successful
        if result is None or len(result) == 0 or result[0] is None:
            print(f"❌ Mesh generation failed!")
            print(f"   This usually means the inference steps were too low or settings were invalid.")
            print(f"   Try increasing num_inference_steps to at least 20.")
            import sys
            sys.exit(1)
        
        mesh = result[0]
        
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

