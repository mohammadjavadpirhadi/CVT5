import os
import argparse
import json
import torch

from dataloaders.compressed_video import *
from models.cvt5 import *
from utils.trainer import move_batch_to_device
from utils.captioning import caption_ids_to_string
from utils.losses import get_loss_fn

parser = argparse.ArgumentParser()

parser.add_argument('--model',
                    required=True,type=str,
                    help='path to model file')
parser.add_argument('--num_beams',
                    required=False,type=int,default=1,
                    help='generate num beams')
parser.add_argument('--max_new_tokens',
                    required=False,type=int,default=50,
                    help='generate max new tokens')
parser.add_argument('--do_sample',
                    required=False,action='store_true',
                    help='generate do sample')
parser.add_argument('--top_k',
                    required=False,type=int,
                    help='generate top k')
parser.add_argument('--top_p',
                    required=False,type=float,
                    help='generate top p')
parser.add_argument('--ignore_blanks',
                    required=False,action='store_true',
                    help='ignore blank predictions')
parser.add_argument('--evaluate',
                    required=False,action='store_true',
                    help='whether to do evaluation')

args = parser.parse_args()

generate_kwargs = {'num_beams': args.num_beams, 'max_new_tokens': args.max_new_tokens}
if args.top_k:
    generate_kwargs['top_k'] = args.top_k
if args.top_p:
    generate_kwargs['top_p'] = args.top_p
if args.do_sample:
    generate_kwargs['do_sample'] = args.do_sample

print("Generate kwargs:", generate_kwargs)

with open(f"{args.model.rsplit('/', 1)[0]}/config.json") as config_file:
    config = json.load(config_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_dataset, test_dataloader = get_dataset_and_data_loader(config["test_dataset"])
model = CVT5Model(config["cvt5_config"]).to(device)
model.load_state_dict(torch.load(args.model), strict=False)

model.train(False)
predictions = {}
with torch.no_grad():
    for i, batch in tqdm(enumerate(test_dataloader), desc=f"Predicting", total=len(test_dataloader)):
        batch = move_batch_to_device(batch, device)
        spotting_logits, caption_logits, _ = model(**batch, **generate_kwargs)
        spotting_predictions = torch.sigmoid(spotting_logits)
        caption_predictions = caption_ids_to_string(caption_logits)
        
        for sample_index in range(spotting_predictions.shape[0]):
            video_index, start_frame_index = batch['indices'][sample_index]
            game_path = test_dataset.get_game_path(video_index)
            if not game_path in predictions:
                predictions[game_path] = {
                    'UrlLocal': game_path,
                    'predictions': []
                }
            middle_point = start_frame_index+1.5*test_dataset.short_memory_len
            position = test_dataset.frame_index_to_position(middle_point)
            events_caption_predictions = caption_predictions[sample_index].split('@')
            event_index = 0
            for label_value, label_index in test_dataset.labels_dict.items():
                if spotting_predictions[sample_index][1][label_index].item() >= 0.5:
                    minute = (position // 1000) // 60
                    second = (position // 1000) % 60
                    half = "1" if "1_224p_h264" in test_dataset.get_frames_path(video_index) else "2"
                    is_blank = not(event_index < len(events_caption_predictions) and events_caption_predictions[event_index].strip() != '')
                    if not (args.ignore_blanks and is_blank):
                        predictions[game_path]['predictions'].append(
                            {
                                "gameTime": f"{half} - {str(minute).rjust(2, '0')}:{str(second).rjust(2, '0')}",
                                "label": label_value,
                                "position": position,
                                "half": half,
                                "confidence": spotting_predictions[sample_index][1][label_index].item(),
                                'comment': events_caption_predictions[event_index] if not is_blank else ''
                            }
                        )
                    event_index += 1
                    # if event_index >= len(events_caption_predictions):
                    #     break

for game_path, predictions_content in tqdm(predictions.items(), desc=f"Writing predictions"):
    output_dir = f"{args.model.rsplit('/', 1)[0]}/outputs/{game_path}/"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/results_dense_captioning.json", "w") as result_file:
        json.dump(predictions_content, result_file)

if args.evaluate:
    from SoccerNet.Evaluation.DenseVideoCaptioning import evaluate as sn_evaluate
    results = sn_evaluate(
        SoccerNet_path=f'{config["test_dataset"]["config"]["base_dir"]}/data/',
        Predictions_path=f"{args.model.rsplit('/', 1)[0]}/outputs/",
        split=config["test_dataset"]["splits"][0],  # Should have one
        version=2,
        prediction_file="results_dense_captioning.json",
        include_SODA=False
    )

