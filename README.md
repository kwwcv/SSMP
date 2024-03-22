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
- Run command `python data/gen_data.py` to generate preprocessed BEV data for training, validating, and testing. Refer to [MotionNet](https://github.com/pxiangwu/MotionNet) for detailed instructions.
- Run command `python data/gen_GSdata.py` to generate preprocessed ground-removed BEV data for training.
## TO BE DONE
## Evaluation
### Trained model
|Ratio|Path|
|---|---|
|1% (semi)|[SSMP1%](https://drive.google.com/file/d/1l7NC4uLapSMGbWeQtk5808gX0jdB4IjG/view?usp=sharing)|
|5% (semi)|[SSMP5%](https://drive.google.com/file/d/1sPdObVITSxPssICARqJLrFsCUV8FLwgd/view?usp=sharing)|
|10% (semi)|[SSMP10%](https://drive.google.com/file/d/127u-LxePHyE8wyAbyrR9stLAN-vxOiQ6/view?usp=sharing)|
## Citation
```
@misc{wang2023semisupervised,
      title={Semi-Supervised Class-Agnostic Motion Prediction with Pseudo Label Regeneration and BEVMix}, 
      author={Kewei Wang and Yizheng Wu and Zhiyu Pan and Xingyi Li and Ke Xian and Zhe Wang and Zhiguo Cao and Guosheng Lin},
      year={2023},
      eprint={2312.08009},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 🍭 Acknowledgement
Our project is based on
[MotionNet](https://github.com/pxiangwu/MotionNet)

The optimal transport solver is adopted from
[Self-Point-Flow](https://github.com/L1bra1/Self-Point-Flow)

### License
This project is licensed under [NTU S-Lab License 1.0](LICENSE) 
