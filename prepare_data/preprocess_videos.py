import os, subprocess, datetime, glob, argparse
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--base_dir',
                    required=True,type=str,
                    help='path to base directory containing data folder')
parser.add_argument('--gop_size',
                    required=False,type=int,
                    help='GOP size',default=-1)
parser.add_argument('--target_fps',
                    required=False,type=int,
                    help='output fps',default=2)

args = parser.parse_args()

LOGS_DIR=f"{args.base_dir}/logs"
os.makedirs(LOGS_DIR, exist_ok=True)
ffmpeg_err_path = f"{LOGS_DIR}/ffmpeg_err_{datetime.datetime.now()}.log"

videos_path = sorted(glob.glob(f"{args.base_dir}/data/*/*/*/*224p.mkv") + glob.glob(f"{args.base_dir}/data/*/*/*/*224p.mp4"))
for video_path in tqdm(videos_path, desc="Preprocessing videos"):
    
    if video_path.rsplit('/', 1)[0].startswith('224p'):
        new_video_path = f"{video_path.rsplit('/', 1)[0]}/1_{video_path.rsplit('/', 1)[1]}"
        os.rename(video_path, new_video_path)
        video_path = new_video_path
    
    if args.gop_size > 0:
        h264_video_path = f"{video_path.rsplit('.', 1)[0]}_h264_gop{args.gop_size}.mp4"
    else:
        h264_video_path = f"{video_path.rsplit('.', 1)[0]}_h264.mp4"
    if os.path.exists(h264_video_path):
        continue

    if args.gop_size > 0:
        s = subprocess.getstatusoutput(f'ffmpeg -y -i "{video_path}" -s 224x224 -c:v libx264 -g {args.gop_size} -keyint_min {args.gop_size} -x264opts scenecut=0 -bf 0 -c:a aac -filter:v fps={args.target_fps} -movflags faststart -f mp4 "{h264_video_path}"')
    else:
        s = subprocess.getstatusoutput(f'ffmpeg -y -i "{video_path}" -s 224x224 -c:v libx264 -x264-params bframes=0 -c:a aac -filter:v fps={args.target_fps} -movflags faststart "{h264_video_path}"')
    if s[0] != 0:
        if os.path.exists(h264_video_path):
            os.remove(h264_video_path)
        with open(ffmpeg_err_path, "a") as ffmpeg_err_file:
            ffmpeg_err_file.write(f"{datetime.datetime.now()}\n")
            ffmpeg_err_file.write(f"{'-'*200}\n")
            ffmpeg_err_file.write(f"{video_path}\n")
            ffmpeg_err_file.write(f"{'-'*200}\n")
            ffmpeg_err_file.write(f"{s[1]}\n")
            ffmpeg_err_file.write(f"\n{'*'*200}\n\n")
        print(f'Failed to convert "{video_path}"! Check "{ffmpeg_err_path}" for more details.')
