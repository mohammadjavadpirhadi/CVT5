import torch
import whisper
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import os, datetime, glob, json, argparse
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--base_dir',
                    required=True,type=str,
                    help='path to base directory containing data folder')
parser.add_argument('--asr_batch_size',
                    required=False,type=int,
                    help='batch size',default=32)

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=args.asr_batch_size,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

LOGS_DIR=f"{args.base_dir}/logs"
os.makedirs(args.base_dir, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

transcribe_err_path = f"{LOGS_DIR}/transcribe_err_{datetime.datetime.now()}.log"

videos_path = sorted(glob.glob(f"{args.base_dir}/data/*/*/*/*224p_h264.mp4"))
for video_path in tqdm(videos_path, desc="Extracting transcript"):
    try:
        video_file_name = video_path.split('/')[-1].rsplit('.', 1)[0]
        transcript_path = f"{video_path.rsplit('.', 1)[0]}_transcript.json"
        if os.path.exists(transcript_path):
            continue
        transcript = pipe(whisper.load_audio(video_path), return_timestamps=True)
        with open(transcript_path, "w") as outfile:
            json.dump(transcript, outfile)
    except Exception as e:
        if os.path.exists(transcript_path):
            os.remove(transcript_path)
        with open(transcribe_err_path, "a") as transcribe_frames_err_file:
            transcribe_frames_err_file.write(f"{datetime.datetime.now()}\n")
            transcribe_frames_err_file.write(f"{'-'*200}\n")
            transcribe_frames_err_file.write(f"{transcript_path}\n")
            transcribe_frames_err_file.write(f"{'-'*200}\n")
            transcribe_frames_err_file.write(f"{str(e)}\n")
            transcribe_frames_err_file.write(f"\n{'*'*200}\n\n")
        print(f'Failed to transcribe "{transcript_path}"! Check "{transcribe_err_path}" for more details.')
