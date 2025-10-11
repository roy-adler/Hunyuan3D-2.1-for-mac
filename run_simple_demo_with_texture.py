#!/usr/bin/env python3
"""
Hunyuan3D-2.1 Demo with Texture Generation for macOS
Generates 3D models from images with PBR textures
"""

import sys
import os
import time

# ============================================================================
# PERFORMANCE SETTINGS - Adjust these for speed vs quality tradeoff
# ============================================================================
# Preset options: 'test', 'minimum', 'fast', 'balanced', 'quality'
QUALITY_PRESET = 'fast'  # Change this to 'minimum', 'fast', 'balanced', or 'quality' for better results

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
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from PIL import Image

# Define quality presets
PRESETS = {
    'test': {
        'num_inference_steps': 12,
        'octree_resolution': 96,
        'guidance_scale': 1.5,
        'num_chunks': 1500,
    },
    'minimum': {
        'num_inference_steps': 15,
        'octree_resolution': 192,
        'guidance_scale': 2.0,
        'num_chunks': 3000,
    },
    'fast': {
        'num_inference_steps': 20,
        'octree_resolution': 256,
        'guidance_scale': 3.0,
        'num_chunks': 4000,
    },
    'balanced': {
        'num_inference_steps': 35,
        'octree_resolution': 320,
        'guidance_scale': 4.0,
        'num_chunks': 6000,
    },
    'quality': {
        'num_inference_steps': 50,
        'octree_resolution': 384,
        'guidance_scale': 5.0,
        'num_chunks': 8000,
    }
}

# Apply preset
if QUALITY_PRESET in PRESETS:
    settings = PRESETS[QUALITY_PRESET]
else:
    print(f"Warning: Unknown preset '{QUALITY_PRESET}', using 'balanced'")
    settings = PRESETS['balanced']

print("="*70)
print("Hunyuan3D-2.1 Demo with Texture Generation - macOS Edition")
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

try:
    # ========================================================================
    # Step 1: Generate Shape
    # ========================================================================
    print("="*70)
    print("STEP 1: GENERATING 3D SHAPE")
    print("="*70)
    print("Loading Hunyuan3D shape generation model...")
    print("(First run will download ~3-4 GB of model weights)")
    print()
    
    dtype = torch.float32 if device.type == "mps" else torch.float16
    
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2.1',
        subfolder='hunyuan3d-dit-v2-1',
        device=device,
        dtype=dtype
    )
    
    print(f"✅ Model loaded successfully!")
    print(f"   Device: {pipeline.device}")
    print(f"   Dtype: {pipeline.dtype}")
    print()
    
    # Generate mesh
    image_path = 'assets/roy-laugh.jpeg'
    output_dir = 'save_dir'
    os.makedirs(output_dir, exist_ok=True)
    
    untextured_mesh_path = os.path.join(output_dir, 'mesh_untextured.glb')
    textured_mesh_path = os.path.join(output_dir, 'mesh_textured.obj')
    
    if os.path.exists(image_path):
        print(f"Processing image: {image_path}")
        
        time_estimates = {
            'test': '5-7 minutes',
            'minimum': '7-9 minutes',
            'fast': '9-11 minutes',
            'balanced': '13-16 minutes',
            'quality': '18-22 minutes'
        }
        est_time = time_estimates.get(QUALITY_PRESET, '10-15 minutes')
        print(f"Generating 3D mesh on {device}... (estimated: {est_time})")
        print()
        
        start_time = time.time()
        
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
            sys.exit(1)
        
        mesh = result[0]
        
        # Save untextured mesh
        mesh.export(untextured_mesh_path)
        shape_time = time.time() - start_time
        print(f"✅ Shape generated in {shape_time:.1f} seconds!")
        print(f"   Untextured mesh saved to: {untextured_mesh_path}")
        print()
        
        # Clean up shape pipeline to free memory
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ========================================================================
        # Step 2: Generate Texture
        # ========================================================================
        print("="*70)
        print("STEP 2: GENERATING PBR TEXTURES")
        print("="*70)
        print("Loading texture generation pipeline...")
        print("(First run will download additional model weights)")
        print()
        
        start_time = time.time()
        
        # Configure texture pipeline
        conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768)
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        
        print(f"Detected device for texture generation: {conf.device}")
        print()
        
        tex_pipeline = Hunyuan3DPaintPipeline(conf)
        
        print("✅ Texture pipeline loaded!")
        print()
        print("Generating textures... (this may take 10-20 minutes)")
        print("The pipeline will:")
        print("  1. Remesh the geometry for optimal UV mapping")
        print("  2. Select optimal camera viewpoints")
        print("  3. Generate multi-view normal and position maps")
        print("  4. Generate PBR textures (albedo, metallic, roughness)")
        print("  5. Enhance textures with RealESRGAN")
        print("  6. Bake textures onto the mesh")
        print()
        
        # Generate textures
        textured_mesh = tex_pipeline(
            mesh_path=untextured_mesh_path,
            image_path=image_path,
            output_mesh_path=textured_mesh_path,
            use_remesh=True,
            save_glb=False
        )
        
        texture_time = time.time() - start_time
        print(f"✅ Textures generated in {texture_time:.1f} seconds!")
        print(f"   Textured mesh saved to: {textured_mesh}")
        print()
        
        # Summary
        print("="*70)
        print("GENERATION COMPLETE!")
        print("="*70)
        print(f"Total time: {shape_time + texture_time:.1f} seconds")
        print()
        print("Output files:")
        print(f"  Untextured mesh: {untextured_mesh_path}")
        print(f"  Textured mesh:   {textured_mesh}")
        print()
        print("You can view the .glb and .obj files in:")
        print("  - https://gltf-viewer.donmccurdy.com/")
        print("  - Blender")
        print("  - Or drag them into a 3D viewer")
        print()
        
    else:
        print(f"❌ Image not found: {image_path}")
        print("Please provide an image file path.")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    import traceback
    traceback.print_exc()
    print()
    print("If you encounter errors, please check:")
    print("  1. All dependencies are installed (see requirements-macos.txt)")
    print("  2. Model weights can be downloaded from Hugging Face")
    print("  3. You have enough disk space (~10GB for models)")

print()
print("="*70)

