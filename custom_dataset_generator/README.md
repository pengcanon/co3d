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
   
   For animated FBX files:
   ```bash
   python custom_dataset_generator/src/generate_dataset.py \
     --model_path custom_dataset_generator/assets/rp_sophia_animated_003_idling_FBX/rp_sophia_animated_003_idling.fbx \
     --output_dir custom_dataset_generator/output \
     --category human_body_4d_03 \
     --annotation_format opencv \
     --num_sequences 5 \
     --num_views 400
   ```

   For static OBJ/GLB files:
   ```bash
   python custom_dataset_generator/src/generate_dataset.py \
     --model_path custom_dataset_generator/assets/rp_fabienne_percy_posed_001_OBJ/rp_fabienne_percy_posed_001_200k.obj \
     --output_dir custom_dataset_generator/output \
     --category human_body_04 \
     --annotation_format opencv \
     --sequence_name sequence_001 \
     --scale_adjustment 0.01 \
    --num_views 400
    ```

   Optional annotation convention:
   - `--annotation_format pytorch3d` (default): writes CO3D-compatible `frame_annotations.jgz`
   - `--annotation_format opencv`: writes OpenCV world-to-camera annotations in `frame_annotations.jgz`
   - `--annotation_format both`: writes PyTorch3D annotations in `frame_annotations.jgz` and OpenCV annotations in `frame_annotations_opencv.jgz`

3. Visualize cameras
   
   For PyTorch3D-format annotations (default):
   ```bash
   python custom_dataset_generator/src/visualize_cameras.py \
     --annotation_path custom_dataset_generator/output/human_body/frame_annotations.jgz \
     --stride 10
   ```

   For OpenCV-format annotations:
   ```bash
   python custom_dataset_generator/src/visualize_cameras.py \
     --annotation_path custom_dataset_generator/output/human_body/frame_annotations_opencv.jgz \
     --viewpoint_format opencv \
     --stride 10
   ```
4. Visualize dataset
   
   For PyTorch3D-format annotations (default):
   ```bash
   python custom_dataset_generator/src/visualize_pointcloud.py \
     --dataset_root "custom_dataset_generator/output" \
     --category "human_body_4d_03" \
     --sequence "frame_000028" \
     --stride 10 \
     --show_cameras
   ```

   For OpenCV-format annotations:
   ```bash
   python custom_dataset_generator/src/visualize_pointcloud.py \
     --dataset_root "custom_dataset_generator/output" \
     --category "human_body_4d_03" \
     --sequence "frame_000028" \
     --stride 10 \
     --show_cameras \
     --viewpoint_format opencv
   ```

5. Compare data statistics
   ```bash
   python custom_dataset_generator/src/compare_stats.py --files \
     "D:\GitHub\co3d\custom_dataset_generator\assets\rp_dennis_posed_004_OBJ\rp_dennis_posed_004_100k.obj" \
     "D:\GitHub\co3d\custom_dataset_generator\assets\rp_posed_00178_29_GLB\rp_posed_00178_29.glb" \
     "D:\GitHub\co3d\custom_dataset_generator\assets\rp_posedplus_00068_18_300k\rp_posedplus_00068_18_300k.glb" \
     "D:\GitHub\co3d\custom_dataset_generator\assets\rp_mei_posed_001_OBJ\rp_mei_posed_001_100k.fbx" \
     "D:\GitHub\co3d\custom_dataset_generator\assets\rp_aliyah_4d_004_dancing_BLD\rp_aliyah_4d_004_dancing_4k.blend" \
     "D:\GitHub\co3d\custom_dataset_generator\assets\rp_fabienne_percy_posed_001_OBJ\rp_fabienne_percy_posed_001_200k.obj" \
     "D:\GitHub\co3d\custom_dataset_generator\assets\rp_manuel_animated_001_dancing_FBX\rp_manuel_animated_001_dancing.fbx" \
     "D:\GitHub\co3d\custom_dataset_generator\assets\rp_nathan_animated_003_walking_FBX\rp_nathan_animated_003_walking.fbx" \
     "D:\GitHub\co3d\custom_dataset_generator\assets\rp_sophia_animated_003_idling_FBX\rp_sophia_animated_003_idling.fbx"
   ```