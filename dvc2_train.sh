#! /bin/bash

source /home/pirhadi/env/bin/activate

python train.py --config dvc2_tv_sml-60_gop-a_Ife-clip_ur-1_ut-1_pfe-resp_a-90.json
python train.py --config dvc2_tv_sml-60_gop-a_Ife-clip_ur-0_ut-1_pfe-resp_a-90.json
python train.py --config dvc2_tv_sml-60_gop-a_Ife-clip_ur-1_ut-0_pfe-resp_a-90.json
python train.py --config dvc2_tv_sml-60_gop-a_Ife-clip_ur-0_ut-0_pfe-resp_a-90.json

python train.py --config dvc2_tv_sml-40_gop-a_Ife-clip_ur-0_ut-1_pfe-resp_a-90.json

python train.py --config dvc2_tv_sml-60_gop-60_Ife-clip_ur-0_ut-1_pfe-resp_a-90.json

python train.py --config dvc2_tv_sml-60_gop-a_Ife-clip_ur-0_ut-1_pfe-resp_a-40.json
python train.py --config dvc2_tv_sml-60_gop-a_Ife-clip_ur-0_ut-1_pfe-resp_a-60.json
