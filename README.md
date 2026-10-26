# Semantic-Guided Density Control for Compact Semantic 3D Gaussian Splatting

Anchor-based instantiation of SGDC, built on top of [ObjectGS](https://github.com/RuijieZhu94/ObjectGS/). The primitive-based instantiation on Gaussian Grouping is available at [SGDC-GS](https://github.com/yuweidong/sgdc-gs).

## Installation

1. Clone the repository locally:

```bash
git clone https://github.com/ZehuiWU2009/sgdc-objectgs
```

2. Install dependencies:
```bash
cd sgdc-objectgs
conda env create --file environment.yml
```

3. Install customized gsplat:

**❗️❗️❗️Attention: You should install the [customized gsplat](https://github.com/RuijieZhu94/gsplat-object), instead of the original gsplat.**

If compilation fails, you will likely need to install `libglm-dev`:

```bash
sudo apt-get install libglm-dev
# then reinstall gsplat-object
pip install git+https://github.com/RuijieZhu94/gsplat-object.git
```

## Preparing the data

- **Download**: You can download the LERF-Masked dataset from this [link](https://huggingface.co/mqye/Gaussian-Grouping/tree/main/data/lerf_mask).

- **Preprocessing**: Please refer to [PLY_PREPROCESSING_README.md](PLY_PREPROCESSING_README.md) for detailed usage.

## Training and Evaluate

### LERF-Mask
#### Ramen scene
```bash
python train.py --config config/objectgs/3d/lerf_mask/config.yaml --scene_name ramen
python render_lerf_mask.py -m path/to/your/training/folder  --skip_train
python eval_lerf_mask.py ramen --pred_path path/to/your/training/folder
python metrics.py -m path/to/your/training/folder
```

#### Figurines scene
```bash
python train.py --config config/objectgs/3d/lerf_mask/config.yaml --scene_name figurines
python render_lerf_mask.py -m path/to/your/training/folder  --skip_train
python eval_lerf_mask.py figurines --pred_path path/to/your/training/folder
python metrics.py -m path/to/your/training/folder
```

#### Teatime scene
```bash
python train.py --config config/objectgs/3d/lerf_mask/config.yaml --scene_name teatime
python render_lerf_mask.py -m path/to/your/training/folder  --skip_train
python eval_lerf_mask.py teatime --pred_path path/to/your/training/folder
python metrics.py -m path/to/your/training/folder
```

### Mip-NeRF 360
```bash
python train.py --config config/objectgs/3d/lerf_mask/config.yaml --scene_name counter
python metrics.py -m path/to/your/training/folder
``` 

## Acknowledgements
This project is built upon [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [Scaffold-GS](https://github.com/city-super/Scaffold-GS) and [ObjectGS](https://github.com/RuijieZhu94/ObjectGS/) . We gratefully acknowledge the authors for open-sourcing their code and making this research possible.
