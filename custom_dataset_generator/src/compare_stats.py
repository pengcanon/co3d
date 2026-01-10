
import os
import sys
import numpy as np
import trimesh
import argparse
import subprocess
import json
import tempfile
from tabulate import tabulate

def get_blender_stats_script():
    return """
import bpy
import sys
import json
import numpy as np
import os

def json_numpy_serializer(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.float32): 
        return float(o)
    raise TypeError

def get_target_center():
    targets = [o for o in bpy.context.scene.objects if o.type == 'MESH' and not o.hide_render]
    
    if not targets:
        return np.array([0., 0., 0.]), np.array([0., 0., 0.]),  np.array([0., 0., 0.])
    
    # Calculate geometric center of bounds from EVALUATED mesh (deformed)
    depsgraph = bpy.context.view_layer.depsgraph
    
    min_pt = np.array([float('inf')] * 3)
    max_pt = np.array([float('-inf')] * 3)
    
    has_valid_mesh = False
    
    for obj in targets:
        if 'floor' in obj.name.lower() or 'plane' in obj.name.lower(): continue 
        
        # Get evaluated object (with modifiers like Armature applied)
        try:
            obj_eval = obj.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh()
        except:
            continue

        if mesh:
            has_valid_mesh = True
            # Transform verts to world space
            # Note: matrix_world is on the object, but we need consistent world coords
            verts = [obj.matrix_world @ v.co for v in mesh.vertices]
            if len(verts) > 0:
                vs = np.array(verts)
                obj_min = np.min(vs, axis=0)
                obj_max = np.max(vs, axis=0)
                min_pt = np.minimum(min_pt, obj_min)
                max_pt = np.maximum(max_pt, obj_max)
            obj_eval.to_mesh_clear()
    
    if not has_valid_mesh:
         return np.array([0., 0., 0.]), np.array([0., 0., 0.]), np.array([0., 0., 0.])

    center = (min_pt + max_pt) / 2.0
    extents = max_pt - min_pt
    
    return min_pt, max_pt, extents

try:
    argv = sys.argv
    if "--" in argv:
        args = argv[argv.index("--") + 1:]
    else:
        args = []
    
    file_path = args[0]
    ext = os.path.splitext(file_path)[1].lower()
    
    # Reset
    bpy.ops.wm.read_homefile(use_empty=True)

    if ext == '.blend':
        bpy.ops.wm.open_mainfile(filepath=file_path)
    elif ext == '.fbx':
        # Import FBX
        # Suppress output if possible, but difficult in Blender
        bpy.ops.import_scene.fbx(filepath=file_path, use_anim=False)
    
    # Process
    min_b, max_b, extents = get_target_center()
    max_dim = float(np.max(extents))
    centroid = (min_b + max_b) / 2.0
    
    stats = {
        "filename": os.path.basename(file_path),
        "min_bound": min_b.tolist(),
        "max_bound": max_b.tolist(),
        "size (extents)": extents.tolist(),
        "max_dimension": max_dim,
        "centroid": centroid.tolist()
    }
    
    print("JSON_START")
    print(json.dumps(stats))
    print("JSON_END")
    
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"""

def analyze_with_blender(file_path, blender_path="blender"):
    print(f"Analyzing with Blender: {os.path.basename(file_path)}...")
    
    # Create temp script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        script_content = get_blender_stats_script()
        tmp.write(script_content)
        script_path = tmp.name
        
    try:
        # Run blender
        # blender -b -P script.py -- file_path
        cmd = [blender_path, "-b", "-P", script_path, "--", file_path]
        
        # Capture output
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"Blender failed: {result.stderr}")
            return None
            
        # Parse JSON from stdout
        output = result.stdout
        start_tag = "JSON_START"
        end_tag = "JSON_END"
        
        if start_tag in output and end_tag in output:
            json_str = output.split(start_tag)[1].split(end_tag)[0].strip()
            return json.loads(json_str)
        else:
            print(f"Could not find JSON output from Blender. Output:\n{output[:200]}...")
            return None
            
    except Exception as e:
        print(f"Error executing Blender: {e}")
        return None
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


def analyze_model_file(model_path, blender_path="blender"):
    """
    Analyzes the original 3D model file (OBJ/GLB/FBX/BLEND) for spatial statistics.
    """
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return None

    # Check extension
    ext = os.path.splitext(model_path)[1].lower()
    
    if ext in ['.fbx', '.blend']:
        return analyze_with_blender(model_path, blender_path)
    
    print(f"Loading model (Trimesh): {os.path.basename(model_path)}...")
    
    # Load model using trimesh directly (bypassing pyrender wrapping for direct analysis)
    # process=False prevents trimesh from auto-centering/scaling immediately
    try:
        scene = trimesh.load(model_path, process=False)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None
    
    bounds = scene.bounds
    centroid = scene.centroid
    extents = scene.extents
    
    # Calculate scale (max dimension)
    max_dim = np.max(extents)
    
    stats = {
        "filename": os.path.basename(model_path),
        "min_bound": bounds[0],
        "max_bound": bounds[1],
        "size (extents)": extents,
        "max_dimension": max_dim,
        "centroid": centroid
    }
    return stats

def main():
    parser = argparse.ArgumentParser(description="Compare statistics of 3D model files")
    parser.add_argument("--files", nargs='+', required=True, help="List of model files to compare")
    parser.add_argument("--blender_path", type=str, default="blender", help="Path to blender executable")
    args = parser.parse_args()

    results = []
    
    print("\n" + "="*80)
    print(" Analyzing Original 3D Model Files ")
    print("="*80 + "\n")

    for file_path in args.files:
        stats = analyze_model_file(file_path, args.blender_path)
        if stats:
            results.append(stats)

    if not results:
        print("No valid models found.")
        return

    # Prepare table data
    table_data = []
    for r in results:
        # Format vectors for display
        c = r['centroid']
        e = r['size (extents)']
        
        centroid_str = f"[{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}]"
        extents_str = f"[{e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f}]"
        
        table_data.append([
            r['filename'],
            f"{r['max_dimension']:.4f}",
            extents_str,
            centroid_str
        ])

    headers = ["Filename", "Max Dim (Scale)", "Extents (X, Y, Z)", "Centroid (X, Y, Z)"]
    
    print("\n")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("\n")
    
    print("Interpretation:")
    for r in results:
        fname = r['filename']
        max_dim = r['max_dimension']
        
        print(f"\n{fname}:")
        if max_dim > 100:
            print(f"  - Appears to be in CENTIMETERS or MILLIMETERS (Scale ~ {max_dim:.1f})")
            print(f"  - ideally should be ~1.8 if representing a human in Meters.")
            print(f"  - Recommendation: Scale by factor of 0.01 (if cm) or 0.001 (if mm).")
        elif max_dim > 10:
             print(f"  - Unusual scale ({max_dim:.1f}). Check units.")
        elif max_dim < 0.5:
             print(f"  - Appears small/toy-scale ({max_dim:.3f}).")
        else:
             print(f"  - Appears to be in METERS (Scale ~ {max_dim:.2f}).")

if __name__ == "__main__":
    main()
