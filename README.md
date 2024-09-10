# Semi-Supervised Class-Agnostic Motion Prediction with Pseudo Label Regeneration and BEVMix
Official implementation for our AAAI2024 paper: "Semi-Supervised Class-Agnostic Motion Prediction with Pseudo Label Regeneration and BEVMix" [**[Arxiv]**](https://arxiv.org/abs/2312.08009)

## 🔨 Dependencies and Installation
- Pytorch >= 1.7.1
```
# git clone this repository
git clone https://github.com/kwwcv/SSMP
cd SSMP
```
### Dataset
- Download the [nuScenes data](https://www.nuscenes.org/).
```
# modified the following paths in gen_data.py, gen_GSdata.py, and data_utils.py
# sys.path.append('root_path/SSMP')
# sys.path.append('root_path/SSMP/nuscenes-devkit/python-sdk/')
```
- Run command `python data/gen_data.py` to generate preprocessed BEV data for training, validating, and testing. Refer to [MotionNet](https://github.com/pxiangwu/MotionNet) and `python data/gen_data.py -h` for detailed instructions.
  
- Install the ground segmentation algorithm following [Patchwork++](https://github.com/url-kaist/patchwork-plusplus). One can also try removing the ground points by simply setting a threshold along the Z-axis.
```
# modified the following path in gen_GSdata.py
# patchwork_module_path = "root_path/patchwork-plusplus/build/python_wrapper"
```
- Run command `python data/gen_GSdata.py` to generate preprocessed ground-removed BEV data for training.
## 🔥 Training
Randomly divide the training data into labeled and unlabeled data sets.
```
# stage 1: train model with only labeled data
python train_stage1.py --data [bev training folder] --seed [random seed] --log

# stage two: train model with both labeled and unlabeled data
python train_stage2.py --data [bev training folder] --GSdata_root [ground removal bev training folder] \
      --resume [stage 1 trained model] --seed [random seed] --if_lr --if_bevmix --log

# Keep the same [random seed] to make sure stage 1 and stage 2 are using the same labeled data set
# when using [random seed] to randomly divide unlabeled and labeled data.
```

One can also use the same divided labeled and unlabeled data sets as used in the paper to train the model.
```
# stage 1
python train_stage1.py --data [bev training folder] --preset_semi [split file] --log

# stage 2
python train_stage2.py --data [bev training folder] --GSdata_root [ground removal bev training folder] \
      --resume [stage 1 trained model] --preset_semi [split file] --if_lf --if_bevmix --log
```

## 🎯 Evaluation
### Trained model
|Ratio|Path|
|---|---|
|1% (semi)|[SSMP1%](https://drive.google.com/file/d/1l7NC4uLapSMGbWeQtk5808gX0jdB4IjG/view?usp=sharing)|
|5% (semi)|[SSMP5%](https://drive.google.com/file/d/1sPdObVITSxPssICARqJLrFsCUV8FLwgd/view?usp=sharing)|
|10% (semi)|[SSMP10%](https://drive.google.com/file/d/127u-LxePHyE8wyAbyrR9stLAN-vxOiQ6/view?usp=sharing)|
### Testing
```
python test.py --data [bev testing folder] --model [model path] \
      --log_path [path to save results] --jitter
```
## Citation
```
@inproceedings{wang2024semi,
  title={Semi-supervised Class-Agnostic Motion Prediction with Pseudo Label Regeneration and BEVMix},
  author={Wang, Kewei and Wu, Yizheng and Pan, Zhiyu and Li, Xingyi and Xian, Ke and Wang, Zhe and Cao, Zhiguo and Lin, Guosheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={5490--5498},
  year={2024}
}
```

## 🍭 Acknowledgement
Our project is based on
[MotionNet](https://github.com/pxiangwu/MotionNet)

The optimal transport solver is adopted from
[Self-Point-Flow](https://github.com/L1bra1/Self-Point-Flow)

### License
This project is licensed under [NTU S-Lab License 1.0](LICENSE) 
