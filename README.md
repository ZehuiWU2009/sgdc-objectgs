<h3 align="center"><strong>ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting</strong></h3>

<p align="center">
  <a href="https://ruijiezhu94.github.io/ruijiezhu/">Ruijie Zhu</a><sup>1,2</sup>,</span>
  <a href="https://mulinyu.github.io/">Mulin Yu</a><sup>2</sup>,</span>
  <a href="https://eveneveno.github.io/lnxu">Linning Xu</a><sup>3</sup>,</span>
  <a href="https://jianglh-whu.github.io/">Lihan Jiang</a><sup>1,2</sup>,</span>
  <a href="https://yixuanli98.github.io/">Yixuan Li</a><sup>3</sup>,</span><br> 
  <a href="https://staff.ustc.edu.cn/~tzzhang/">Tianzhu Zhang</a><sup>1</sup>,</span>
  <a href="https://oceanpang.github.io/">Jiangmiao Pang</a><sup>2</sup>,</span>
  <a href="https://daibo.info/">Bo Dai</a><sup>4</sup></span>
  <br>
  <sup>1</sup> USTC </span> 
  <sup>2</sup> Shanghai AI Lab </span> 
  <sup>3</sup> CUHK </span> 
  <sup>4</sup> HKU </span>
  <br>
  <b>ICCV 2025</b>
</p>

<div align="center">
  <a href='https://arxiv.org/abs/2507.15454'><img src='https://img.shields.io/badge/arXiv-2507.15454-b31b1b.svg'></a>      
  <a href='https://ruijiezhu94.github.io/ObjectGS_page/'><img src='https://img.shields.io/badge/Project-Page-orange'></a>      
  <!-- <a href='https://ruijiezhu94.github.io/ObjectGS_page/'><img src='https://img.shields.io/badge/YouTube-Demo-yellow'></a>        -->
  <a href='https://github.com/RuijieZhu94/ObjectGS?tab=MIT-1-ov-file'><img src='https://img.shields.io/badge/License-MIT-green'></a>      
  <a href='https://ruijiezhu94.github.io/ObjectGS'><img src="https://visitor-badge.laobi.icu/badge?page_id=ruijiezhu94.objectgs"/></a>
  <br>
  <br>
</div>

<br>

<p align="center">
<img src="assets/pipeline.jpg" width="97%"/>
</p>

> The overall architecture of ObjectGS. We first use a 2D segmentation pipeline to assign object ID and lift it to 3D. Then we initialize the anchors and use them to generate object-aware neural Gaussians. To provide semantic guidance, we model the Gaussian semantics and construct classification-based constraints. As a result, our method enables both object-level and scene-level reconstruction.

## 🚀 Quick Start

### 🔧 Dataset Preparation

To train ObjectGS, you should download the following dataset:

* [3DOVS](https://github.com/Kunhao-Liu/3D-OVS)
* [LERF-Mask](https://github.com/lkeab/gaussian-grouping/blob/main/docs/dataset.md#1-lerf-mask-dataset)
* [Replica](https://github.com/facebookresearch/Replica-Dataset)
* [Scannet++](https://github.com/scannetpp/scannetpp)

Or you can use our processed subsets:
[Google Drive](https://drive.google.com/drive/folders/1rKeDAactXtjqb37PiJ68ia-KyoKehUmy?usp=sharing)

We organize the datasets as follows:

```shell
├── data
│   | 3dovs
│     ├── bed
│     ├── bench
│     ├── ...
│   | lerf_mask
│     ├── figurines
│     ├── ramen
│     ├── teatime
│   | replica
│     ├── office_0
│     ├── office_1
│     ├── ...
│   | scannetpp_ovs
│     ├── 09bced689e
│     ├── 0d2ee665be
│     ├── ...
```

### 🛠️ Installation

1. Clone this repo:

```bash
git clone git@github.com:RuijieZhu94/ObjectGS.git
```

2. Install dependencies:

```bash
cd ObjectGS
conda env create --file environment.yml
```

**❗️❗️❗️Attention: You should install my [customized gsplat](https://github.com/RuijieZhu94/gsplat-object), instead of the original gsplat.**

If compilation fails, you will likely need to install `libglm-dev`:

```bash
sudo apt-get install libglm-dev
# then reinstall gsplat-object
pip install git+https://github.com/RuijieZhu94/gsplat-object.git
```

### 🌟 Training

#### Data Preprocessing

Please refer to [PLY_PREPROCESSING_README.md](PLY_PREPROCESSING_README.md) for detailed usage.

#### Model Training

```shell
# 3dgs version
bash train_3d.sh /path/to/your/dataset
# 2dgs version
bash train_2d.sh /path/to/your/dataset
```

### 🎇 Evaluation

#### Rendering 2D Semantics

```shell
bash render.sh path/to/your/training/folder
```

#### Rendering 3D Semantics (Point Cloud)

```shell
python vis_ply.py
```

#### Rendering single object

```shell
bash render_object.sh path/to/your/training/folder
```

Specify `query_label_id` (0-255) to render single object. -1 means loop through rendering all single objects.

#### Open-Vocabulary Segmentation Evaluation

Please refer to [Gaussian-Grouping](https://github.com/lkeab/gaussian-grouping/blob/main/docs/dataset.md#2-render-mask-with-text-prompt) for OVS evaluation.

#### Exporting Mesh

```shell
# exporting scene mesh
python export_mesh.py -m path/to/your/training/folder

# exporting object mesh
python export_object_mesh.py -m path/to/your/training/folder --query_label_id -1
## query_label_id: specify a object id (0~255), -1 for all objects
```

### 📜 Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{zhu2025objectgs,
  title={ObjectGS: Object-aware Scene Reconstruction and Scene Understanding via Gaussian Splatting},
  author={Zhu, Ruijie and Yu, Mulin and Xu, Linning and Jiang, Lihan and Li, Yixuan and Zhang, Tianzhu and Pang, Jiangmiao and Dai, Bo},
  booktitle={International Conference on Computer Vision (ICCV), 2025},
  year={2025}
}
```

### 🤝 Acknowledgements

Our code is based on [Scaffold-GS](https://city-super.github.io/scaffold-gs/), [HorizonGS](https://city-super.github.io/horizon-gs/), [GSplat](https://github.com/nerfstudio-project/gsplat) and [Gaussian-Grouping](https://github.com/lkeab/gaussian-grouping). We thank the authors for their excellent work!

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export all_proxy=socks5://127.0.0.1:7897  # 如果是 SOCKS5 代理

python train.py --config config/objectgs/3d/lerf_mask/config.yaml --scene_name ramen

python render_lerf_mask.py -m outputs/lerf_mask/rebuttal/ramen/2026-08-28_14:32:58  --skip_train

python eval_lerf_mask.py ramen --pred_path outputs/lerf_mask/rebuttal/ramen/2026-08-28_14:32:58/test/ours_30000_text/test_mask

python metrics.py -m outputs/replica/objectgs/office_0/2026-09-08_15:43:36
