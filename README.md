# Fabric Material Recovery from Video Using Multi-Scale Geometric Auto-Encoder

Junbang Liang, Ming C. Lin. ECCV 2022.

[Project Page](https://gamma.umd.edu/researchdirections/virtualtryon/fabric_estimation)

### Requirements
- Python 3.6.4
- [PyTorch](https://pytorch.org/) tested on version 1.1.0
- trimesh
- opencv-python
- smplx
- chumpy

### Train

- If you wish to train the garment geometry auto-encoder:
```
python train.py --train_module atlas
```
- If you wish to train the single-frame body and garment estimation module:
```
python train.py --train_module single
```
- If you wish to train the garment material estimation module:
```
python train.py --train_module material
```

### Data
Training/test dataset, as well as pretrained model, can be found at [here](). Please unzip them and place them under the folder of this repo (same parent folder as this file).

### Demo
- If you wish to test the single-frame reconstruction only:
```
python realworld_demo.py --img ${your_test_image_path}
```
- If you wish to test the material estimation:
```
python video_demo.py --img ${path_of_splitted_video_frame_images}
```

### Citation
If you use this code for your research, please consider citing:
```
@inProceedings{liang2022fabric,
  title={Fabric Material Recovery from Video Using Multi-Scale Geometric Auto-Encoder},
  author = {Junbang Liang and Ming C. Lin},
  booktitle={ECCV},
  year={2022}
}
```

### Acknowledgement
This code repository is based on [SPIN](https://github.com/nkolot/SPIN).
