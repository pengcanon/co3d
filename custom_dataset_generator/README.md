# Custom CO3D Dataset Generator

This folder contains tools to generate a CO3D-compatible dataset from a 3D model (e.g., `.glb` format).

## Structure

- `assets/`: Place your `.glb` model here.
- `assets/tex/`: Place your texture files here.
- `src/`: Python scripts for generation.
- `output/`: The generated dataset will appear here.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the generation script:
   ```bash
   python custom_dataset_generator/src/generate_dataset.py --model_path custom_dataset_generator/assets/rp_posedplus_00068_18_300k/rp_posedplus_00068_18_300k.glb --output_dir custom_dataset_generator/output --category human_body --sequence_name sequence_001 --scale_adjustment 0.01
   ```

3. Visualize cameras
   ```bash
   python custom_dataset_generator/src/visualize_cameras.py --annotation_path custom_dataset_generator/output/human_body/frame_annotations.jgz --stride 10
   ```       

4. Compare data statistics
   ```bash
   python custom_dataset_generator/src/compare_stats.py --files "D:\GitHub\co3d\custom_dataset_generator\assets\rp_dennis_posed_004_OBJ\rp_dennis_posed_004_100k.obj" "D:\GitHub\co3d\custom_dataset_generator\assets\rp_posed_00178_29_GLB\rp_posed_00178_29.glb" "D:\GitHub\co3d\custom_dataset_generator\assets\rp_posedplus_00068_18_300k\rp_posedplus_00068_18_300k.glb"       
   ```    
   