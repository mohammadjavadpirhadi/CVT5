import torch
from transformers import UMT5EncoderModel, AutoTokenizer

import os, datetime, glob, json, argparse
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--base_dir',
                    required=True,type=str,
                    help='path to base directory containing data folder')
parser.add_argument('--chunk_size_s',
                    required=False,type=int,
                    help='output chunk size',default=30)

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_id = "google/umt5-base"

model = UMT5EncoderModel.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

LOGS_DIR=f"{args.base_dir}/logs"
os.makedirs(args.base_dir, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

feature_extraction_err_path = f"{LOGS_DIR}/asr_feature_extraction_err_{datetime.datetime.now()}.log"

model.train(False)

def merge_chunks(transcript, chunk_size):
    text = ''
    current_chunk_index = 0
    chunks_merged = {
        "indices": [],
        "texts": [],
    }
    for chunk in transcript["chunks"]:
        # Sometimes last chunk timestamp is broken
        if chunk["timestamp"][0] is not None and chunk["timestamp"][0]//chunk_size != current_chunk_index:
            chunks_merged["indices"].append(current_chunk_index)
            chunks_merged["texts"].append(text)
            text = chunk["text"]
            current_chunk_index = chunk["timestamp"][0]//chunk_size
        else:
            text += chunk["text"]

    chunks_merged["indices"].append(current_chunk_index)
    chunks_merged["texts"].append(text)
    return chunks_merged

transcripts_path = glob.glob(f"{args.base_dir}/data/*/*/*/*_transcript.json")
with torch.no_grad():
    for transcript_path in tqdm(transcripts_path, desc="Extracting ASR features"):
        try:
            features_path = f"{transcript_path.rsplit('.', 1)[0]}_features_{args.chunk_size_s}.pt"
            if os.path.exists(features_path):
                continue
            with open(transcript_path, "r") as json_transript:
                transcript = merge_chunks(json.load(json_transript), args.chunk_size_s)
            features = {}
            for chunk_index, text in zip(transcript["indices"], transcript["texts"]):
                input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
                outputs = model(input_ids=input_ids)
                features[int(chunk_index)] = outputs.last_hidden_state.squeeze().detach().cpu()
            torch.save(features, features_path)
        except Exception as e:
            if os.path.exists(features_path):
                os.remove(features_path)
            with open(feature_extraction_err_path, "a") as feature_extraction_err_file:
                feature_extraction_err_file.write(f"{datetime.datetime.now()}\n")
                feature_extraction_err_file.write(f"{'-'*200}\n")
                feature_extraction_err_file.write(f"{transcript_path}\n")
                feature_extraction_err_file.write(f"{'-'*200}\n")
                feature_extraction_err_file.write(f"{str(e)}\n")
                feature_extraction_err_file.write(f"\n{'*'*200}\n\n")
            print(f'Failed to extract features "{transcript_path}"! Check "{feature_extraction_err_path}" for more details.')
