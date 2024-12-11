from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time, datetime
import os
import torch

from compressed_video import CompressedVideoDataset, get_dataset_and_data_loader

config = {
    "splits": [
        "train",
        "valid",
        "test"
    ],
    "config": {
        "stage": 1,
        "task": "caption",
        "base_dir": "/home/pirhadi",
        "mv_shape": 56,
        "res_shape": 56,
        "short_memory_len": 60,
        "gop_size": -1,
        "feature_extrator": "clip",
        "use_residuals": True,
        "use_transcripts_features": True,
        "transcripts_features_shape": 768,
        "fps": 2,
        "pad_videos": True,
        "add_bg_label": False,
        "batch_size": 4,
        "overlap_strategy": "no",
        "max_samples_per_epoch": 20,
        # "pos_neg_prob_ratio": 3,
        "shuffle": False,
        "load_labels": True
    }
}

dataset, dataloader = get_dataset_and_data_loader(config)
# dataset = CompressedVideoDataset(config["splits"], config["config"])
# start = datetime.datetime.now()
# print(dataset.get_bce_loss_pos_weights())
# print(datetime.datetime.now() - start)

x = []
for iter_index in tqdm(dataset.iter_indices, total=len(dataset)):
    video_index, start_frame_index = iter_index[0], iter_index[1]
    spotting_labels = dataset.spotting_labels[video_index][start_frame_index+dataset.short_memory_len:start_frame_index+2*dataset.short_memory_len]
    x.append(spotting_labels.sum().item())
    # print(spotting_labels.shape)
    # print('-'*20)
# print(x)
print(len(x), x.count(0.0), len(x)-x.count(0.0))

# x = 0
# for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
    # print(batch['captioner_attention_masks'].shape, batch['captioner_attention_masks'])
    # print(batch['transcripts_features'].shape, batch['transcripts_features'])
    # print(batch['spotting_labels'].shape)
    # print(torch.sum(batch['spotting_labels'][:,1,:], dim=-1).shape, torch.sum(batch['spotting_labels'][:,1,:], dim=-1))
    # input()
    # if batch['spotting_labels'][0][dataset.short_memory_len].sum().item()>0:
    #     print("FUCK YOU")
    #     x += 1
    # print(batch['spotting_labels'][0][dataset.short_memory_len], batch['spotting_labels'][0][dataset.short_memory_len].sum().item(), batch['spotting_labels'][0][dataset.short_memory_len].sum().item()>0)
    # input()
# print(x)

# for spotting_label in dataset.spotting_labels:
#     print(spotting_label.shape)
#     spotting_label_sum = torch.sum(spotting_label, dim=-1)
#     print(spotting_label[torch.nonzero(spotting_label_sum, as_tuple=True)[0]])
#     print(torch.nonzero(spotting_label_sum, as_tuple=True), torch.nonzero(spotting_label_sum).shape)
#     print(spotting_label_sum.shape)
#     raise

# print(dataset.iter_indices.shape)
# x = []
# print(x)
# print(len(x))
# print(x.count(0.0))
# print(1-x.count(0.0)/len(x))
# def calculate_sample_weights(dataset):
#     weights = torch.zeros((dataset.iter_indices.shape[0]))
#     for i, iter_index in enumerate(dataset.iter_indices):
#         video_index, start_frame_index = iter_index[0], iter_index[1]
#         spotting_labels = dataset.spotting_labels[video_index][start_frame_index+dataset.short_memory_len:start_frame_index+2*dataset.short_memory_len]
#         weights[i] = spotting_labels.sum().item()
#     weights[weights==0] = weights.sum() / (weights == 0.0).sum()
#     return weights
# print(calculate_sample_weights(dataset))

# dataloader = DataLoader(
#     bas_dataset_train,
#     batch_size=2,
#     num_workers=os.cpu_count(),
#     shuffle=False,
#     collate_fn=compressed_video_collate_fn,
# )

# for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
#     # print('before process')
#     # print(batch)
#     # input()
#     # time.sleep(5)
#     # print('after process')
#     # raise
#     pass

# print(bas_dataset_train.get_iter_len())
# # input("GO")
# keys = ['frames_map', 'I_frames', 'temporal_masks', 'spotting_labels', 'captions_input_ids'] if bas_config["load_labels"] else ['frames_map', 'temporal_masks']
# for epoch in range(1):
#     print(f'EPOCH: {epoch}')
#     for batch in bas_dataset_train:
#         # print(batch['captions'])
#         for key in keys:
#             print(key, batch[key].shape, batch[key])
#         i += 1
#         if i == 3:
#             break
# print(bas_dataset_train.get_bce_loss_pos_weights().shape, bas_dataset_train.get_bce_loss_pos_weights())
# print(bas_dataset_train.get_focal_loss_alpha().shape, bas_dataset_train.get_focal_loss_alpha())
