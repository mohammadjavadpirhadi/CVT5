#! /bin/bash

source /home/pirhadi/env/bin/activate

# python train.py --config dvc1_tv_sml-60_gop-a_Ife-clip_ur-1_ut-1_pfe-resp_a-90.json
# python train.py --config dvc1_tv_sml-60_gop-a_Ife-clip_ur-0_ut-1_pfe-resp_a-90.json
# Since using transcripts has no effect on first stage, we can skip following trainings.
# python train.py --config dvc1_tv_sml-60_gop-a_Ife-clip_ur-1_ut-0_pfe-resp_a-90.json
# python train.py --config dvc1_tv_sml-60_gop-a_Ife-clip_ur-0_ut-0_pfe-resp_a-90.json

# python train.py --config dvc1_tv_sml-40_gop-a_Ife-clip_ur-0_ut-1_pfe-resp_a-90.json

python train.py --config dvc1_tv_sml-60_gop-60_Ife-clip_ur-0_ut-1_pfe-resp_a-90.json
