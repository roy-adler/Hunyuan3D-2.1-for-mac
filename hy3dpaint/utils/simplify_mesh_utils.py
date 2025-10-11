# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import trimesh
import pymeshlab


def remesh_mesh(mesh_path, remesh_path):
    mesh = mesh_simplify_trimesh(mesh_path, remesh_path)


def mesh_simplify_trimesh(inputpath, outputpath, target_count=40000):
    # 先去除离散面
    ms = pymeshlab.MeshSet()
    if inputpath.endswith(".glb"):
        ms.load_new_mesh(inputpath, load_in_a_single_layer=True)
    else:
        ms.load_new_mesh(inputpath)
    ms.save_current_mesh(outputpath.replace(".glb", ".obj"), save_textures=False)
    # 调用减面函数
    courent = trimesh.load(outputpath.replace(".glb", ".obj"), force="mesh")
    face_num = courent.faces.shape[0]

    print(f"Mesh simplification: Current faces={face_num}, Target={target_count}")
    
    # Only simplify if we have significantly more faces than target
    if face_num > target_count * 1.5:
        # Ensure target_count is valid (at least 4 faces, max 90% of original)
        safe_target = max(4, min(target_count, int(face_num * 0.9)))
        print(f"Simplifying mesh from {face_num} to {safe_target} faces...")
        try:
            courent = courent.simplify_quadric_decimation(safe_target)
            actual_faces = courent.faces.shape[0]
            print(f"✅ Mesh simplified to {actual_faces} faces")
        except Exception as e:
            print(f"⚠️  Simplification failed: {e}")
            print(f"   Using original mesh with {face_num} faces")
    else:
        print(f"Mesh face count ({face_num}) is reasonable, skipping simplification")
    
    courent.export(outputpath)
