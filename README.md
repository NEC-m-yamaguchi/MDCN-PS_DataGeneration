# MDCN-PS: Monocular-Depth-guided Coarse Normal attention for Robust Photometric Stereo (WACV2025)

Abstract: Photometric Stereo (PS) is a technique for estimating surface normals from images illuminated by multiple light sources. However, when the target object has a complex shape or the light sources are not appropriately arranged, certain regions may experience severe shadows, leading to insufficient information for accurate estimation. In this paper, we propose a Monocular-Depth-guided Coarse Normal attention for Photometric Stereo (MDCN-PS). The MDCN-PS can effectively combine monocular depth from a single image with PS with multiple light sources by a Photometric Stereo network Adaptor (PS Adaptor) with Coarse Normal Attention. The key is to use the coarse normals obtained from Monocular Depth Estimation as supplementary information, which can improve accuracy in regions where the light source is limited due to severe shadows or inhomogeneous light source distribution. Comprehensive experiments on real-world and synthetic datasets show that the proposed method achieved an accuracy improvement of 1.2 points in real-world datasets when limited to two input images and of 3.1 points in synthetic datasets in mean angular error compared to existing methods. Qualitative results also demonstrated that our method improves accuracy in areas with insufficient lighting patterns due to shadows. 

The following is shared here.
- Code to generate the dataset for training
- Model and Material lists for generating dataset

## Prerequisites
- Python3
- blender3.5
- cv2
- objaverse

## Installation
1. Install `objaverse` via pip 
   ```bash
   pip install objaverse
   ```
1. Install Blender and make sure the `bpy` module is available.
1. Install `cv2` (OpenCV) in blender python via pip:
   ```bash
   pip install opencv-python
   ```

## Data Generation
1. Download the models in madels_list.txt to `assets/Models/` using the following      command.
   ```bash
   python load_models.py
   ```
1. Additionally download models from [blob dataset](https://people.csail.mit.edu/kimo/blobs/) to `assets/models/`.
1. Download the materials in materials_list.txt to `assets/Materials/` from [ambientCG](https://ambientcg.com/).
1. Using the following command to generate the dataset.
   ```bash
   python processor.py --object_folder_path ./assets/Models/  --material_folder_path ./assets/Materials/ --output_folder_path /path/to/output/foleders/ --num_scenes num_of_generate_scenes
   ```

## Notes
- Our method used the above data generation code to create 30488 scenes and used them as training data.

## Citation

If you find our work useful in your research, please consider citing:

    @inproceedings{yamaguchi2025mdcn,
        author = {Yamaguchi, Masahiro and Shibata, Takashi and Yachida, Shoji and Yokoyama Keiko and Hosoi, Toshinori},
        title = {MDCN-PS: Monocular-Depth-guided Coarse Normal attention for Robust Photometric Stereo},
        booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
        year = {2025}
    }
