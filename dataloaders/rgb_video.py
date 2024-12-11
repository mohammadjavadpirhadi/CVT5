import os
import glob
import json
from tqdm.auto import tqdm
import PIL.Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from SoccerNet.utils import getListGames
from transformers import AutoTokenizer

from dataloaders.constants import *

umt5_tokenizer = AutoTokenizer.from_pretrained('google/umt5-base')

class RGBVideoDataset(Dataset):

    def __init__(self, splits, config):
        self.stage = config['stage']
        assert self.stage in [1, 2], "Stage should be one of 1 or 2"
        self.task = config['task']
        if self.task == "spotting-ball":
            self.labels_dict = BALL_ACTION_SPOTTING_LABELS
            self.labels_file_name = "Labels-ball.json"
        elif self.task == "caption":
            self.labels_dict = DENSE_CAPTIONING_LABELS
            self.labels_file_name = "Labels-caption.json"
        elif self.task == "spotting":
            self.labels_dict = ACTION_SPOTTING_LABELS
            self.labels_file_name = "Labels-v2.json"
        self.base_dir = config['base_dir']
        self.short_memory_len = config['short_memory_len']
        self.use_transcripts_features = config['use_transcripts_features']
        if self.use_transcripts_features:
            self.transcripts_features_shape = config['transcripts_features_shape']
        self.fps = config['fps']
        self.pad_videos = config['pad_videos']
        self.add_bg_label = config['add_bg_label']
        self.game_limit = config['game_limit'] if 'game_limit' in config else None
        assert config['overlap_strategy'] in ['no', 'half', 'quarter', 'all'], "Overlap stratgey should be one of 'no', 'half', 'quarter' or 'all'"
        if config['overlap_strategy'] == 'no':
            self.iter_step_size = self.short_memory_len
        elif config['overlap_strategy'] == 'half':
            assert self.short_memory_len % 2 == 0, "Short memory len should be even for using 'half' overlap strategy."
            self.iter_step_size = self.short_memory_len // 2
        elif config['overlap_strategy'] == 'quarter':
            assert self.short_memory_len % 4 == 0, "Short memory len should be dividable by 4 for using 'quarter' overlap strategy."
            self.iter_step_size = self.short_memory_len // 4
        else:
            self.iter_step_size = 1
        self.shuffle = config['shuffle']
        self.load_labels = config['load_labels']
        
        self.build_frames_pathes(splits)
        self.load_frames()
        if self.use_transcripts_features:
            self.load_transcripts_features()
        if self.load_labels:
            self.build_lables()
        self.build_iter_indices()

        print(f"Dataset info:")
        print(f"Splits: {splits}")
        print(f"Config: {config}")
        print(f"Total possible iterations: {self.iter_indices.shape[0]}")

    def position_to_frame_index(self, position):
        frame_index = int(position / 1000 * self.fps)

        if self.pad_videos:
            frame_index += self.short_memory_len

        return frame_index

    def frame_index_to_position(self, frame_index):
        if self.pad_videos:
            frame_index -= self.short_memory_len

        return int(frame_index / self.fps * 1000)

    def get_game_path(self, video_index):
        return self.game_pathes[video_index]

    def build_frames_pathes(self, splits):
        self.frames_pathes = []
        self.game_pathes = []

        for split in tqdm(splits, desc="Building frames pathes"):
            game_list = getListGames(split=split, task=self.task)
            if self.game_limit is not None:
                game_list = game_list[:self.game_limit]
            for game_path in game_list:
                game_frames_pathes = glob.glob(f"{self.base_dir}/data/{game_path}/*_h264_rgb")
                self.frames_pathes += game_frames_pathes
                self.game_pathes += [game_path]*len(game_frames_pathes)

    def load_frames(self):
        self.frames_map = []
        self.temporal_masks = []
        self.I_frames = []

        for frames_path in tqdm(self.frames_pathes, desc="Scanning frames pathes"):
            video_frames_map = []
            video_I_frames = []

            if self.pad_videos:
                video_frames_map += [FRAMES_MAP['<PAD>'] for _ in range(self.short_memory_len)]

            frame_pathes = sorted(glob.glob(f"{frames_path}/*.pt"))
            for frame_path in tqdm(frame_pathes, desc="Frames", position=1, leave=False):
                video_I_frames.append(frame_path)
                video_frames_map.append(FRAMES_MAP['I'])

            if self.pad_videos:
                if len(video_frames_map) % self.short_memory_len != 0:
                    video_frames_map += [FRAMES_MAP['<PAD>'] for _ in range(self.short_memory_len - len(video_frames_map) % self.short_memory_len)]
                video_frames_map += [FRAMES_MAP['<PAD>'] for _ in range(self.short_memory_len)]

            self.frames_map.append(torch.tensor(video_frames_map, dtype=torch.int))
            self.temporal_masks.append(self.frames_map[-1] == FRAMES_MAP['<PAD>'])
            self.I_frames.append(video_I_frames)

    def load_transcripts_features(self):
        self.transcripts_features = []
        for frames_path in tqdm(self.frames_pathes, desc="Loading transcripts features"):
            transcript_features_path = f"{frames_path}_transcript_features_{self.short_memory_len//self.fps}.pt"
            if os.path.exists(transcript_features_path):
                self.transcripts_features.append(torch.load(transcript_features_path))
            else:
                self.transcripts_features.append(None)

    def build_lables(self):
        self.spotting_labels = []
        self.captions = []
        self.fps_list = []

        for i, frames_path in enumerate(tqdm(self.frames_pathes, desc="Building labels")):
            label_path = f"{frames_path[:-1].rsplit('/', 1)[0]}/{self.labels_file_name}"
            if not os.path.exists(label_path):
                continue
            
            video_spotting_labels = torch.zeros((self.get_video_len(i), len(self.labels_dict)), dtype=torch.float32)
            video_captions = {frame_index:[] for frame_index in range(self.get_video_len(i))}

            with open(label_path) as label_file:
                annotations = json.load(label_file)['annotations']
            
            half = frames_path[:-1].rsplit('/', 1)[1].split('_', 1)[0]
            for annotation in tqdm(annotations, desc="Annotations", position=1, leave=False):
                if not annotation['gameTime'].startswith(half):
                    continue
                if not annotation['label'] in self.labels_dict:
                    continue
                frame_position = self.position_to_frame_index(int(annotation['position']))
                if not (0 <= frame_position < video_spotting_labels.shape[0]):
                    continue
                
                video_spotting_labels[
                    frame_position,
                    self.labels_dict[annotation['label']]
                ] = 1

                caption_position = int(video_spotting_labels[frame_position, :self.labels_dict[annotation['label']]].sum().item())
                if "anonymized" in annotation:
                    video_captions[frame_position].insert(caption_position, annotation["anonymized"])
                else:
                    video_captions[frame_position].insert(caption_position, annotation["label"])
            video_captions = {frame_position:'@'.join(captions) for frame_position, captions in video_captions.items()}

            if self.add_bg_label:
                video_spotting_bg_labels = video_spotting_labels.count_nonzero(dim=1)
                video_spotting_bg_labels[video_spotting_bg_labels > 0] = 1
                video_spotting_bg_labels = 1 - video_spotting_bg_labels
                self.spotting_labels.append(torch.cat((video_spotting_bg_labels.unsqueeze(dim=1), video_spotting_labels), dim=1))
            else:
                self.spotting_labels.append(video_spotting_labels)
            self.captions.append(video_captions)
    
    def build_iter_indices(self):
        iter_indices = []
        video_iter_lengths = []
        video_index_masks = []
        for video_index in range(len(self.frames_pathes)):
            video_iter_indices = [(video_index, start_frame_index) for start_frame_index in range(0, self.get_video_len(video_index)-3*self.short_memory_len+1, self.iter_step_size) if (self.stage == 1 or self.spotting_labels[video_index][start_frame_index+self.short_memory_len:start_frame_index+2*self.short_memory_len].sum() > 0)]
            iter_indices += video_iter_indices
            video_iter_lengths.append(len(video_iter_indices))
            video_index_masks += [video_index] * video_iter_lengths[-1]
        self.iter_indices = torch.LongTensor(iter_indices)
        self.video_iter_lengths = torch.LongTensor(video_iter_lengths)
        self.video_index_masks = torch.LongTensor(video_index_masks)
        self.chosen_times = torch.zeros((len(self.iter_indices)), dtype=torch.long)

    def get_video_len(self, idx):
        return len(self.frames_map[idx])

    def get_frames_path(self, idx):
        return self.frames_pathes[idx]
    
    def build_caption(self, video_index, frames_start_index):
        caption = ''
        for frame_offset in range(self.short_memory_len):
            frame_index = frames_start_index.item()+self.short_memory_len+frame_offset
            if self.captions[video_index][frame_index] != '':
                caption += f'{self.captions[video_index][frame_index]}@'
        caption = caption.rsplit('@', 1)[0]
        return caption
        #return self.captions[video_index.item()][(frames_start_index+self.short_memory_len).item()]

    def get_chunk(self, video_index, frames_start_index):
        I_frames_start_index = torch.sum(self.frames_map[video_index][:frames_start_index] == FRAMES_MAP['I'])
        P_frames_start_index = torch.sum(self.frames_map[video_index][:frames_start_index] == FRAMES_MAP['P'])
        I_frames_count = torch.sum(self.frames_map[video_index][frames_start_index:frames_start_index+3*self.short_memory_len] == FRAMES_MAP['I'])
        P_frames_count = torch.sum(self.frames_map[video_index][frames_start_index:frames_start_index+3*self.short_memory_len] == FRAMES_MAP['P'])

        if self.load_labels:
            spotting_labels = torch.stack(
                [
                    self.spotting_labels[video_index][frames_start_index:frames_start_index+self.short_memory_len].max(dim=0).values,
                    self.spotting_labels[video_index][frames_start_index+self.short_memory_len:frames_start_index+2*self.short_memory_len].max(dim=0).values,
                    self.spotting_labels[video_index][frames_start_index+2*self.short_memory_len:].max(dim=0).values
                ],
                dim=0
            )

        if self.use_transcripts_features:
            if self.transcripts_features[video_index] is not None and frames_start_index.item()//self.short_memory_len in self.transcripts_features[video_index]:
                transcripts_features = self.transcripts_features[video_index][frames_start_index.item()//self.short_memory_len]
            else:
                transcripts_features = torch.zeros((0, self.transcripts_features_shape))
        else:
            transcripts_features = None
        
        return {
            'index': (video_index, frames_start_index),
            'frames_map': self.frames_map[video_index][frames_start_index:frames_start_index+3*self.short_memory_len],
            'temporal_mask': self.temporal_masks[video_index][frames_start_index:frames_start_index+3*self.short_memory_len],
            'I_frames': torch.stack([torch.load(I_frame) for I_frame in self.I_frames[video_index][I_frames_start_index:I_frames_start_index+I_frames_count]], dim=0),
            'transcripts_features': transcripts_features,
            'spotting_labels': spotting_labels if self.load_labels else None,
            'caption': self.build_caption(video_index, frames_start_index) if self.load_labels else None
        }

    def __getitem__(self, index):
        return self.get_chunk(self.iter_indices[index][0], self.iter_indices[index][1])
    
    def __len__(self):
        return len(self.iter_indices)

    def get_bce_loss_pos_weights(self, power: float = 1.0, reduction='none'):
        all_labels = []
        for i, iter_index in enumerate(self.iter_indices):
            video_index, start_frame_index = iter_index[0], iter_index[1]
            spotting_labels = self.spotting_labels[video_index][start_frame_index+self.short_memory_len:start_frame_index+2*self.short_memory_len].max(dim=0).values
            all_labels.append(spotting_labels)
        all_labels = torch.stack(all_labels, dim=0)
        positive_count = all_labels.count_nonzero(dim=0)
        negative_count = all_labels.shape[0] - positive_count
        pos_weight = torch.nan_to_num(negative_count / positive_count, nan=0.0, posinf=0.0, neginf=0.0)
        
        pos_weight **= power

        if reduction == 'sum':
            pos_weight = pos_weight.sum()
        elif reduction == 'mean':
            pos_weight = pos_weight.mean()
        return pos_weight

    def get_focal_loss_alpha(self):
        all_labels = []
        for i, iter_index in enumerate(self.iter_indices):
            video_index, start_frame_index = iter_index[0], iter_index[1]
            spotting_labels = self.spotting_labels[video_index][start_frame_index+self.short_memory_len:start_frame_index+2*self.short_memory_len].max(dim=0).values
            all_labels.append(spotting_labels)
        all_labels = torch.stack(all_labels, dim=0)
        return (1 - all_labels.count_nonzero() / all_labels.numel())


def rgb_video_collate_fn(batch):
    indices = []
    frames_map = []
    temporal_masks = []
    captioner_attention_masks = []
    I_frames = []
    use_transcripts_features = batch[0]['transcripts_features'] is not None
    if use_transcripts_features:
        transcripts_features = []
    has_labels = batch[0]['spotting_labels'] is not None
    if has_labels:
        spotting_labels = []
        captions = []
    transcripts_features_max_len = max(batch[i]["transcripts_features"].shape[0] if batch[i]["transcripts_features"] is not None else 0 for i in range(len(batch)))
    transcripts_features_count = max(batch[i]["transcripts_features"].shape[1] if batch[i]["transcripts_features"] is not None else 0 for i in range(len(batch)))
    for sample in batch:
        indices.append(sample["index"])
        frames_map.append(sample['frames_map'])
        temporal_masks.append(sample['temporal_mask'])
        captioner_attention_masks.append(~sample['temporal_mask']) # HuggingFace masks have exactly the opposite behaviour in comparison to PyTorch
        I_frames.append(sample['I_frames'])
        if use_transcripts_features:
            if sample['transcripts_features'] is not None:
                transcripts_features.append(
                    torch.cat(
                        [
                            sample['transcripts_features'],
                            torch.zeros((transcripts_features_max_len-sample['transcripts_features'].shape[0], transcripts_features_count))
                        ],
                        dim=0
                    )
                )
                captioner_attention_masks[-1] = torch.cat(
                    [
                        captioner_attention_masks[-1],
                        torch.cat(
                            [
                                torch.ones((sample['transcripts_features'].shape[0])),
                                torch.zeros((transcripts_features_max_len-sample['transcripts_features'].shape[0]))
                            ],
                            dim=0
                        )
                    ],
                    dim=0
                )
            else:
                transcripts_features.append(
                    torch.zeros((transcripts_features_max_len, transcripts_features_count))
                )
                captioner_attention_masks[-1] = torch.cat(
                    [
                        captioner_attention_masks[-1],
                        torch.zeros((transcripts_features_max_len))
                    ],
                    dim=0
                )

        if has_labels:
            spotting_labels.append(sample['spotting_labels'])
            captions.append(sample['caption'])
    
    if has_labels:
        captions_input_ids = umt5_tokenizer(captions, padding="longest", truncation=True, max_length=512, return_tensors="pt").input_ids
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        captions_input_ids[captions_input_ids == umt5_tokenizer.pad_token_id] = -100

    return {
        'indices': indices,
        'frames_map': torch.stack(frames_map, dim=0),
        'temporal_masks': torch.stack(temporal_masks, dim=0),
        'captioner_attention_masks': torch.stack(captioner_attention_masks, dim=0),
        'I_frames': torch.cat(I_frames, dim=0),
        'transcripts_features': torch.stack(transcripts_features, dim=0) if use_transcripts_features else None,
        'spotting_labels': torch.stack(spotting_labels, dim=0) if has_labels else None,
        'captions': captions if has_labels else None,
        'captions_input_ids': captions_input_ids if has_labels else None
    }

def calculate_sample_weights(dataset, pos_neg_prob_ratio):
    all_labels = []
    for i, iter_index in enumerate(dataset.iter_indices):
        video_index, start_frame_index = iter_index[0], iter_index[1]
        spotting_labels = dataset.spotting_labels[video_index][start_frame_index+dataset.short_memory_len:start_frame_index+2*dataset.short_memory_len].max(dim=0).values
        all_labels.append(spotting_labels)
    all_labels = torch.stack(all_labels, dim=0)
    positive_count = all_labels.count_nonzero(dim=0)
    total_positive_count = positive_count.sum()
    pos_weights = (total_positive_count - positive_count) / total_positive_count

    weights = torch.zeros((dataset.iter_indices.shape[0]))
    for i, iter_index in enumerate(dataset.iter_indices):
        # video_index, start_frame_index = iter_index[0], iter_index[1]
        # spotting_labels = dataset.spotting_labels[video_index][start_frame_index+dataset.short_memory_len:start_frame_index+2*dataset.short_memory_len].max(dim=0).values
        # weights[i] = spotting_labels.sum().item()
        weights[i] = (pos_weights*all_labels[i]).sum().item()
    # This way the probability of all non action windows will be equal to all action windows 
    weights[weights==0] = (weights.sum() / pos_neg_prob_ratio) / (weights == 0.0).count_nonzero()
    return weights

def get_dataset_and_data_loader(config):
    dataset = RGBVideoDataset(config["splits"], config["config"])
    if dataset.load_labels:
        max_samples_per_epoch = min(config["config"]["max_samples_per_epoch"] if "max_samples_per_epoch" in config["config"] else len(dataset), len(dataset))
        pos_neg_prob_ratio = config["config"]["pos_neg_prob_ratio"] if "pos_neg_prob_ratio" in config["config"] else 1.0
        sample_weights = calculate_sample_weights(dataset, pos_neg_prob_ratio)
        data_loader = DataLoader(
            dataset,
            batch_size=config["config"]["batch_size"],
            # num_workers=os.cpu_count(),
            num_workers=24,
            # shuffle=config["config"]["shuffle"],
            collate_fn=rgb_video_collate_fn,
            sampler=WeightedRandomSampler(weights=sample_weights, num_samples=max_samples_per_epoch, replacement=False, generator=None) if config["config"]["shuffle"] else None
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=config["config"]["batch_size"],
            # num_workers=os.cpu_count(),
            num_workers=24,
            shuffle=config["config"]["shuffle"],
            collate_fn=rgb_video_collate_fn,
        )
    return dataset, data_loader
