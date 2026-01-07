
import os
import sys
import numpy as np
import trimesh
import argparse
from tabulate import tabulate

def analyze_model_file(model_path):
    """
    Analyzes the original 3D model file (OBJ/GLB) for spatial statistics.
    """
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return None

    print(f"Loading model: {os.path.basename(model_path)}...")
    
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
    args = parser.parse_args()

    results = []
    
    print("\n" + "="*80)
    print(" Analyzing Original 3D Model Files ")
    print("="*80 + "\n")

    for file_path in args.files:
        stats = analyze_model_file(file_path)
        if stats:
            results.append(stats)

    if not results:
        print("No valid models found.")
        return

    # Prepare table data
    table_data = []
    for r in results:
        # Format vectors for display
        centroid_str = f"[{r['centroid'][0]:.2f}, {r['centroid'][1]:.2f}, {r['centroid'][2]:.2f}]"
        extents_str = f"[{r['size (extents)'][0]:.2f}, {r['size (extents)'][1]:.2f}, {r['size (extents)'][2]:.2f}]"
        
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
