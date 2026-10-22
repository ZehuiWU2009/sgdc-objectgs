

python train.py --config config/objectgs/3d/lerf_mask/config.yaml --scene_name ramen

python render_lerf_mask.py -m outputs/lerf_mask/rebuttal/ramen/2026-08-28_14:32:58  --skip_train

python eval_lerf_mask.py ramen --pred_path outputs/lerf_mask/rebuttal/ramen/2026-08-28_14:32:58/test/ours_30000_text/test_mask

python metrics.py -m outputs/replica/objectgs/office_0/2026-09-08_15:43:36
