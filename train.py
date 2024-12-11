import os
import argparse
import json
import random
import math
import torch
import numpy as np

GLOBAL_SEED = 2024

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.use_deterministic_algorithms(True)

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from transformers import get_cosine_schedule_with_warmup

from dataloaders.compressed_video import get_dataset_and_data_loader
from models.cvt5 import *
from utils.losses import get_loss_fn
from utils.trainer import train_one_epoch, evaluate_one_epoch, evaluate_one_epoch_sn

parser = argparse.ArgumentParser()

parser.add_argument('--config',
                    required=True,type=str,
                    help='path to config file')
parser.add_argument('--last_checkpoint',
                    required=False,type=str,
                    help='last checkpoint to continue from')

args = parser.parse_args()

with open(f"{os.getcwd()}/configs/{args.config}") as config_file:
    config = json.load(config_file)

do_validation = "valid_dataset" in config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"Number of CPU cores: {os.cpu_count()}")

train_dataset, train_dataloader = get_dataset_and_data_loader(config["train_dataset"])
if do_validation:
    valid_dataset, valid_dataloader = get_dataset_and_data_loader(config["valid_dataset"])
    test_dataset, test_dataloader = get_dataset_and_data_loader(config["test_dataset"])
spottting_loss_fn = get_loss_fn(train_dataset, config["spotting_loss_fn"])
model = CVT5Model(config["cvt5_config"], spottting_loss_fn).to(device)

if "pretrained_model" in config:
    print(f"Loading model weights from {config['pretrained_model']['path']}...")
    loaded_state_dict = torch.load(config["pretrained_model"]["path"])
    for exception in config["pretrained_model"]["exceptions"]:
        loaded_state_dict.pop(exception)
    model.load_state_dict(loaded_state_dict, strict=False)

    for param in model.parameters():
        param.requires_grad = not config["pretrained_model"]["freeze"]["enabled"]
    for module_name in config["pretrained_model"]["freeze"]["exceptions"]:
        for param in getattr(model, module_name).parameters():
            param.requires_grad = config["pretrained_model"]["freeze"]["enabled"]

num_params = sum(p.numel() for p in model.parameters())
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params:,d} (Trainable: {num_trainable_params:,d})")

optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
num_cycles = config["num_cosine_schedule_cycles"] if "num_cosine_schedule_cycles" in config else 0.5
scheduler = get_cosine_schedule_with_warmup(optimizer, config["num_warmup_steps"], config["epoches"]*len(train_dataloader), num_cycles=num_cycles)

if args.last_checkpoint:
    if args.last_checkpoint[-1] == '/':
        args.last_checkpoint = args.last_checkpoint[:-1]
    print(f"Resuming from {args.last_checkpoint}...")
    timestamp = args.last_checkpoint.rsplit('/', 1)[1]    
    model.load_state_dict(torch.load(f'{args.last_checkpoint}/model_last.pt'))
    optimizer.load_state_dict(torch.load(f'{args.last_checkpoint}/optimizer_last.pt'))
    scheduler.load_state_dict(torch.load(f'{args.last_checkpoint}/scheduler_last.pt'))
else:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter(f'{config["base_dir"]}/runs/{args.config.rsplit(".", 1)[0]}_{timestamp}')
STATES_DIR = f'{config["base_dir"]}/{config["name"]}_states/{timestamp}'
os.makedirs(STATES_DIR, exist_ok=True)
with open(f"{STATES_DIR}/config.json", "w") as config_file:
    json.dump(config, config_file)

if do_validation:
    if args.last_checkpoint:
        import glob
        from tensorboard.backend.event_processing import event_accumulator

        past_losses = []
        past_meteors = []
        past_epochs = []
        for event_path in glob.glob(f"{writer.log_dir}/*events*"):
            past_events = event_accumulator.EventAccumulator(event_path)
            past_events.Reload()
            if 'Loss/valid' in past_events.Tags()['scalars']:
                past_losses += [scaler.value for scaler in past_events.Scalars('Loss/valid')]
            if 'meteor/valid sn' in past_events.Tags()['scalars']:
                past_meteors += [scaler.value for scaler in past_events.Scalars('meteor/valid sn')]
                past_epochs += [scaler.step for scaler in past_events.Scalars('meteor/valid sn')]
        best_meteor = max(past_meteors)
        best_loss = min(past_losses)
        start_epoch = max(past_epochs) + 1
        print(f"Retrieved last epoch ({start_epoch}), best meteor ({best_meteor}) and best loss ({best_loss})!")
    else:
        best_meteor = 0.0
        best_loss = 999999999999
        start_epoch = 0
else:
    start_epoch = 0

for epoch in range(start_epoch, config["epoches"]):
    print('EPOCH {}:'.format(epoch + 1))

    model.train(True)
    train_one_epoch(
        model,
        train_dataloader,
        config["cvt5_config"]["short_memory_len"],
        config["cvt5_config"]["umt5_enabled"],
        int(config["train_dataset"]["config"]["add_bg_label"]),
        epoch,
        optimizer,
        scheduler,
        writer,
        device,
        config["verbose"]
    )
    if device == 'cuda':
        torch.cuda.empty_cache()

    if do_validation:
        model.train(False)
        with torch.no_grad():
            average_loss, _, _, _, _, _ = evaluate_one_epoch(
                model,
                valid_dataloader,
                config["cvt5_config"]["short_memory_len"],
                config["cvt5_config"]["umt5_enabled"],
                int(config["valid_dataset"]["config"]["add_bg_label"]),
                epoch,
                writer,
                device,
                config["verbose"]
            )
            
            average_meteor_valid_sn = evaluate_one_epoch_sn(
                model,
                test_dataloader,
                test_dataset,
                config["test_dataset"]["splits"][0], # Should have one split
                config["cvt5_config"]["umt5_enabled"],
                epoch,
                config["test_dataset"]["config"]["base_dir"],
                STATES_DIR,
                writer,
                device
            )

    writer.flush()

    if config["save_model"]:
        torch.save(model.state_dict(), f'{STATES_DIR}/model_last.pt')
        torch.save(optimizer.state_dict(), f'{STATES_DIR}/optimizer_last.pt')
        torch.save(scheduler.state_dict(), f'{STATES_DIR}/scheduler_last.pt')

        if do_validation:
            if average_meteor_valid_sn >= best_meteor:
                best_meteor = average_meteor_valid_sn
                torch.save(model.state_dict(), f'{STATES_DIR}/model_best.pt')
                torch.save(optimizer.state_dict(), f'{STATES_DIR}/optimizer_best.pt')
                torch.save(scheduler.state_dict(), f'{STATES_DIR}/scheduler_best.pt')
            if average_loss <= best_loss:
                best_loss = average_loss
                torch.save(model.state_dict(), f'{STATES_DIR}/model_es.pt')
                torch.save(optimizer.state_dict(), f'{STATES_DIR}/optimizer_es.pt')
                torch.save(scheduler.state_dict(), f'{STATES_DIR}/scheduler_es.pt')

writer.close()

