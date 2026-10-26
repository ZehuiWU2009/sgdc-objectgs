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


## Training and Evaluate


python train.py --config config/objectgs/3d/lerf_mask/config.yaml --scene_name ramen

python render_lerf_mask.py -m outputs/lerf_mask/rebuttal/ramen/2026-08-28_14:32:58  --skip_train

python eval_lerf_mask.py ramen --pred_path outputs/lerf_mask/rebuttal/ramen/2026-08-28_14:32:58/test/ours_30000_text/test_mask

python metrics.py -m outputs/replica/objectgs/office_0/2026-09-08_15:43:36
