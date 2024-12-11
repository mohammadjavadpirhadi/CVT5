import os, shutil, datetime, glob, math, cv_reader, flow_vis, cv2, decord, argparse
import numpy as np
from tqdm.auto import tqdm

decord.bridge.set_bridge("torch")

parser = argparse.ArgumentParser()

parser.add_argument('--base_dir',
                    required=True,type=str,
                    help='path to base directory containing data folder')
parser.add_argument('--gop_size',
                    required=False,type=int,
                    help='GOP size',default=-1)

args = parser.parse_args()

LOGS_DIR=f"{args.base_dir}/logs"
os.makedirs(LOGS_DIR, exist_ok=True)
extract_frames_err_path = f"{LOGS_DIR}/extract_frames_err_{datetime.datetime.now()}.log"

print(f'Searching for games in "{args.base_dir}/data/"...')
if args.gop_size > 0:
    h264_videos_path = glob.glob(f"{args.base_dir}/data/*/*/*/*h264_gop{args.gop_size}.mp4")
else:
    h264_videos_path = glob.glob(f"{args.base_dir}/data/*/*/*/*h264.mp4")

def extract_frames(video_path, output_path):
    decord_reader = decord.VideoReader(video_path)
    video_frames = cv_reader.read_video(video_path=video_path, with_residual=True)

    assert len(video_frames), f'"{video_path}" is empty!'

    os.makedirs(output_path, exist_ok=True)
    num_digits = math.ceil(math.log10(len(video_frames)))

    gop_index = -1
    for frame_idx, frame in tqdm(enumerate(video_frames), total=len(video_frames), desc="Frames", position=1, leave=False):
        if frame['pict_type'] == "I":
            gop_index += 1
            os.makedirs(f"{output_path}/{str(gop_index).rjust(num_digits, '0')}", exist_ok=True)
            cv2.imwrite(
                f"{output_path}/{str(gop_index).rjust(num_digits, '0')}/{str(frame_idx).rjust(num_digits, '0')}_{frame['pict_type']}.jpg",
                cv2.cvtColor(np.array(decord_reader.get_batch([frame_idx]))[0], cv2.COLOR_RGB2BGR) # CV2 assumes image BGR
            )
        else:
            cv2.imwrite(
                f"{output_path}/{str(gop_index).rjust(num_digits, '0')}/{str(frame_idx).rjust(num_digits, '0')}_{frame['pict_type']}_mv.png", # save as png to avoid lossy compression
                cv2.cvtColor(flow_vis.flow_to_color(frame["motion_vector"][..., :2]), cv2.COLOR_RGB2BGR) # As we only use P frames we just need to save pre motion vectors, CV2 assumes image BGR
            )
            cv2.imwrite(
                f"{output_path}/{str(gop_index).rjust(num_digits, '0')}/{str(frame_idx).rjust(num_digits, '0')}_{frame['pict_type']}_res.jpg",
                cv2.cvtColor(frame["residual"], cv2.COLOR_RGB2BGR) # CV2 assumes image BGR
            )

print("Extracting I/P frames:")
for h264_video_path in tqdm(h264_videos_path, desc="Videos"):
    try:
        output_dir = f"{h264_video_path.rsplit('.', 1)[0]}"
        if os.path.exists(output_dir):
            continue
        extract_frames(h264_video_path, output_dir)
    except Exception as e:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        with open(extract_frames_err_path, "a") as extract_frames_err_file:
            extract_frames_err_file.write(f"{datetime.datetime.now()}\n")
            extract_frames_err_file.write(f"{'-'*200}\n")
            extract_frames_err_file.write(f"{h264_video_path}\n")
            extract_frames_err_file.write(f"{'-'*200}\n")
            extract_frames_err_file.write(f"{str(e)}\n")
            extract_frames_err_file.write(f"\n{'*'*200}\n\n")
        print(f'Failed to extract frames from "{h264_video_path}"! Check "{extract_frames_err_path}" for more details.')
