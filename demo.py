#!/usr/bin/env python3
"""
Hunyuan3D-2.1 Demo for macOS
Generates 3D models from images with optional texture generation
"""

import sys
import os
import time
import argparse

# ============================================================================
# Parse Arguments
# ============================================================================
parser = argparse.ArgumentParser(description='Hunyuan3D-2.1 3D Generation Demo')
parser.add_argument('--image', type=str, default='assets/demo.png',
                    help='Path to input image (default: assets/demo.png)')
parser.add_argument('--output', type=str, default='output_mesh.glb',
                    help='Path to output mesh (default: output_mesh.glb)')
parser.add_argument('--texture', action='store_true',
                    help='Enable texture generation (PBR materials)')
parser.add_argument('--no-texture', dest='texture', action='store_false',
                    help='Disable texture generation (geometry only)')
parser.add_argument('--quality', type=str, default='fast',
                    choices=['test', 'minimum', 'fast', 'balanced', 'quality'],
                    help='Quality preset (default: fast)')
parser.add_argument('--steps', type=int, default=None,
                    help='Number of inference steps (overrides preset)')
parser.add_argument('--resolution', type=int, default=None,
                    help='Octree resolution (overrides preset)')
parser.add_argument('--guidance', type=float, default=None,
                    help='Guidance scale (overrides preset)')
parser.add_argument('--chunks', type=int, default=None,
                    help='Number of chunks (overrides preset)')
parser.add_argument('--texture-views', type=int, default=8,
                    help='Number of views for texture generation (default: 8)')
parser.add_argument('--texture-resolution', type=int, default=768,
                    help='Texture resolution (default: 768)')

parser.set_defaults(texture=False)
args = parser.parse_args()

# ============================================================================
# Setup
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
settings = PRESETS[args.quality]

# Override with manual settings if provided
if args.steps is not None:
    settings['num_inference_steps'] = args.steps
if args.resolution is not None:
    settings['octree_resolution'] = args.resolution
if args.guidance is not None:
    settings['guidance_scale'] = args.guidance
if args.chunks is not None:
    settings['num_chunks'] = args.chunks

print("="*70)
print("Hunyuan3D-2.1 Demo - macOS Edition")
print("="*70)
print(f"Input image: {args.image}")
print(f"Output mesh: {args.output}")
print(f"Texture generation: {'ENABLED' if args.texture else 'DISABLED'}")
print(f"Quality preset: {args.quality.upper()}")
print(f"  Inference steps: {settings['num_inference_steps']}")
print(f"  Octree resolution: {settings['octree_resolution']}")
print(f"  Guidance scale: {settings['guidance_scale']}")
print(f"  Memory chunks: {settings['num_chunks']}")
if args.texture:
    print(f"  Texture views: {args.texture_views}")
    print(f"  Texture resolution: {args.texture_resolution}")
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
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        sys.exit(1)
    
    print(f"Processing image: {args.image}")
    
    time_estimates = {
        'test': '5-7 minutes',
        'minimum': '7-9 minutes',
        'fast': '9-11 minutes',
        'balanced': '13-16 minutes',
        'quality': '18-22 minutes'
    }
    est_time = time_estimates.get(args.quality, '10-15 minutes')
    print(f"Generating 3D mesh on {device}... (estimated: {est_time})")
    print()
    
    start_time = time.time()
    
    # Generate with custom settings
    image = Image.open(args.image)
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
        print(f"   Try increasing --steps to at least 20.")
        sys.exit(1)
    
    mesh = result[0]
    
    # Save untextured mesh
    if args.texture:
        # Save to temp location for texture pipeline
        output_dir = 'save_dir'
        os.makedirs(output_dir, exist_ok=True)
        untextured_path = os.path.join(output_dir, 'mesh_untextured.glb')
        mesh.export(untextured_path)
    else:
        # Save directly to output
        mesh.export(args.output)
    
    shape_time = time.time() - start_time
    print(f"✅ Shape generated in {shape_time:.1f} seconds!")
    if not args.texture:
        print(f"   Mesh saved to: {args.output}")
    print()
    
    # Clean up shape pipeline to free memory
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ========================================================================
    # Step 2: Generate Texture (Optional)
    # ========================================================================
    if args.texture:
        print("="*70)
        print("STEP 2: GENERATING PBR TEXTURES")
        print("="*70)
        print("Loading texture generation pipeline...")
        print("(First run will download additional model weights)")
        print()
        
        from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
        
        start_time = time.time()
        
        # Configure texture pipeline
        conf = Hunyuan3DPaintConfig(
            max_num_view=args.texture_views,
            resolution=args.texture_resolution
        )
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
        
        # Determine output format
        if args.output.endswith('.glb'):
            textured_output = args.output.replace('.glb', '_textured.obj')
        elif args.output.endswith('.obj'):
            textured_output = args.output
        else:
            textured_output = args.output + '_textured.obj'
        
        # Generate textures
        textured_mesh = tex_pipeline(
            mesh_path=untextured_path,
            image_path=args.image,
            output_mesh_path=textured_output,
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
        print(f"  Untextured mesh: {untextured_path}")
        print(f"  Textured mesh:   {textured_mesh}")
        print()
    else:
        # Summary for geometry-only
        print("="*70)
        print("GENERATION COMPLETE!")
        print("="*70)
        print(f"Total time: {shape_time:.1f} seconds")
        print(f"Output mesh: {args.output}")
        print()
    
    print("You can view the files in:")
    print("  - https://gltf-viewer.donmccurdy.com/ (for .glb)")
    print("  - https://3dviewer.net/ (for .obj)")
    print("  - Blender")
    print()
        
except Exception as e:
    print(f"❌ Error: {e}")
    print()
    import traceback
    traceback.print_exc()
    print()
    print("If you encounter errors, please check:")
    print("  1. All dependencies are installed: pip install -r requirements-macos.txt")
    print("  2. Model weights can be downloaded from Hugging Face")
    print("  3. You have enough disk space (~10GB for models)")
    sys.exit(1)

print("="*70)
