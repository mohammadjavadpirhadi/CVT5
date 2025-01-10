import os
import math
import glob
import argparse
import torch
from transformers import AutoProcessor, CLIPVisionModel
from tqdm.auto import tqdm

import decord
decord.bridge.set_bridge('torch')

parser = argparse.ArgumentParser()

parser.add_argument('--base_dir',
                    required=True,type=str,
                    help='path to base directory containing data folder')
parser.add_argument('--batch_size',
                    required=False,type=int,
                    help='batch size',default=32)

args = parser.parse_args()

model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.train(False)

video_pathes = glob.glob(f"{args.base_dir}/data/*/*/*/*224p_h264.mp4")
with torch.no_grad():
    for video_path in tqdm(video_pathes):
        output_dir = f"{video_path.rsplit('.', 1)[0]}_rgb"
        os.makedirs(output_dir, exist_ok=True)
        decord_reader = decord.VideoReader(video_path)
        num_digits = math.ceil(math.log10(len(decord_reader)))
        for start_frame_index in tqdm(range(0, len(decord_reader), args.batch_size), position=1, leave=False):
            frames = decord_reader.get_batch(range(start_frame_index, min(start_frame_index+args.batch_size, len(decord_reader))))
            batch = processor(images=frames, return_tensors="pt").pixel_values.to("cuda")
            batch_features = model(batch).pooler_output.squeeze()
            for sample_index in range(batch_features.shape[0]):
                frame_features_path = f"{output_dir}/{str(start_frame_index+sample_index).rjust(num_digits, '0')}.pt"
                torch.save(batch_features[sample_index].detach().cpu(), frame_features_path)

