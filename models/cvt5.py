import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm
from transformers import UMT5ForConditionalGeneration, ViTConfig, ViTModel
from transformers.modeling_outputs import BaseModelOutput
import torchvision.models

from dataloaders.constants import *

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float = 0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class FrameTypeEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float = 0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor, frames_map: Tensor) -> Tensor:
        x = x + self.embeddings(frames_map)
        return self.dropout(x)

class PeriodTypeEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float = 0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor, frames_map: Tensor) -> Tensor:
        x = x + self.embeddings(frames_map)
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, max_pos: int, dropout: float = 0):
        super().__init__()
        self.model_type = 'Transformer'
        self.positional_encoding = PositionalEncoding(d_model, max_pos, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        encoder_norm = LayerNorm(d_model, eps=1e-5, bias=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, norm=encoder_norm)
        self.d_model = d_model

    def forward(self, src: Tensor, src_mask: Tensor = None, src_key_padding_mask: Tensor = None) -> Tensor:
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output

class CVT5Model(nn.Module):

    def __init__(self, config, spotting_loss_fn=None):
        super(CVT5Model, self).__init__()
        self.short_memory_len = config['short_memory_len']
        self.feature_extrator = config['feature_extrator']
        assert self.feature_extrator in ['resnet-pooler', 'resnet-last', 'clip-pooler', 'clip-last'], "Feature extractor should be one of 'resnet-pooler', 'resnet-last', 'clip-pooler' or 'clip-last'"
        self.use_residuals = config['use_residuals']
        self.use_transcripts_features = config['use_transcripts_features']
        self.two_stage_encoder = config["two_stage_encoder"]
        self.umt5_enabled = config['umt5_enabled']

        self.motion_vectors_dropout = nn.Dropout2d(p=config['dropout'])
        if self.use_residuals:
            self.residuals_dropout = nn.Dropout2d(p=config['dropout'])
        
        if self.feature_extrator == 'resnet-pooler':
            self.motion_vectors_encoder = nn.Sequential(*list(torchvision.models.resnet18().children())[:-1])
            if self.use_residuals:
                self.residuals_encoder = nn.Sequential(*list(torchvision.models.resnet18().children())[:-1])
        elif self.feature_extrator == 'resnet-last':
            self.motion_vectors_encoder = nn.Sequential(*list(torchvision.models.resnet18().children())[:-2])
            if self.use_residuals:
                self.residuals_encoder = nn.Sequential(*list(torchvision.models.resnet18().children())[:-2])
        else: # clip-pooler and clip-last
            self.motion_vectors_encoder = ViTModel(ViTConfig(hidden_size=config['mv_hidden_size'], intermediate_size=config['mv_intermediate_size'], num_hidden_layers=config['mv_num_hidden_layers'], image_size=config['mv_shape'], patch_size=config['mv_patch_size']))
            if self.use_residuals:
                self.residuals_encoder = ViTModel(ViTConfig(hidden_size=config['res_hidden_size'], intermediate_size=config['res_intermediate_size'], num_hidden_layers=config['res_num_hidden_layers'], image_size=config['res_shape'], patch_size=config['res_patch_size']))
        
        self.motion_vectors_encoder_dropout = nn.Dropout(p=config['dropout'])
        if self.use_residuals:
            self.residuals_encoder_dropout = nn.Dropout(p=config['dropout'])

        if self.use_residuals:
            self.P_frames_projector = nn.Linear(config['res_feature_shape']+config['mv_feature_shape'], config['I_feature_shape'])
        else:
            self.P_frames_projector = nn.Linear(config['mv_feature_shape'], config['I_feature_shape'])

        self.frames_encoded_dropout = nn.Dropout1d(p=config['dropout'])
        self.frame_type_encoding = FrameTypeEncoding(config['I_feature_shape'], len(FRAMES_MAP))

        self.period_type_encoding = PeriodTypeEncoding(config['I_feature_shape'], 3)
        if self.two_stage_encoder:
            self.period_encoder = TransformerModel(config['I_feature_shape'], config['period_encoder_nhead'], config['period_encoder_d_hid'], config['period_encoder_nlayers'], self.short_memory_len)
            self.period_encoder_dropout = nn.Dropout1d(p=config['dropout'])
        
        self.video_encoder = TransformerModel(config['I_feature_shape'], config['video_encoder_nhead'], config['video_encoder_d_hid'], config['video_encoder_nlayers'], self.short_memory_len*3)
        self.video_encoder_dropout = nn.Dropout(p=config['dropout'])

        self.spotting_head = nn.Linear(config['I_feature_shape'], config['num_classes'])
        self.spotting_loss_fn = spotting_loss_fn
        self.spotting_loss_weight = config['spotting_loss_weight'] if 'spotting_loss_weight' in config else 1.0

        if self.umt5_enabled:
            if self.use_transcripts_features:
                self.transcripts_features_dropout = nn.Dropout1d(p=config['dropout'])
            self.umt5_loss_weight = config['umt5_loss_weight'] if 'umt5_loss_weight' in config else 1.0
            self.umt5_model = UMT5ForConditionalGeneration.from_pretrained('google/umt5-base')
            self.umt5_projector = nn.Linear(config['I_feature_shape'], self.umt5_model.config.d_model)
            if config["freeze_umt5"]:
                for param in self.umt5_model.parameters():
                    param.requires_grad = False
        
        print("CVT5 model info:")
        print(f"Config: {config}")

    def forward(
            self,
            indices,
            frames_map,
            temporal_masks,
            captioner_attention_masks,
            I_frames,
            motion_vectors,
            residuals=None,
            transcripts_features=None,
            spotting_labels=None,
            captions_input_ids=None,
            captions=None,
            **kwargs
        ):
        motion_vectors = self.motion_vectors_dropout(motion_vectors)
        if self.use_residuals:
            residuals = self.residuals_dropout(residuals)

        mv_res_batch_size = motion_vectors.shape[0]
        if self.feature_extrator in ['resnet-pooler', 'resnet-last']:
            motion_vectors_encoded = self.motion_vectors_encoder(motion_vectors).reshape(mv_res_batch_size, -1)
            if self.use_residuals:
                residuals_encoded = self.residuals_encoder(residuals).reshape(mv_res_batch_size, -1)
        elif self.feature_extrator == 'clip-pooler':
            motion_vectors_encoded = self.motion_vectors_encoder(motion_vectors).pooler_output
            if self.use_residuals:
                residuals_encoded = self.residuals_encoder(residuals).pooler_output
        else: # clip-last
            motion_vectors_encoded = self.motion_vectors_encoder(motion_vectors).last_hidden_state[:, 1:, :].reshape(mv_res_batch_size, -1)
            if self.use_residuals:
                residuals_encoded = self.residuals_encoder(residuals).last_hidden_state[:, 1:, :].reshape(mv_res_batch_size, -1)

        motion_vectors_encoded = self.motion_vectors_encoder_dropout(motion_vectors_encoded)
        if self.use_residuals:
            residuals_encoded = self.residuals_encoder_dropout(residuals_encoded)

        if self.use_residuals:
            P_frames_encoded = self.P_frames_projector(
                torch.cat((motion_vectors_encoded, residuals_encoded), dim=1)
            )
        else:
            P_frames_encoded = self.P_frames_projector(
                motion_vectors_encoded
            )

        frames_encoded = []
        I_frame_index = 0
        P_frame_index = 0
        for sample_frames_map in frames_map:
            sample_frames_encoded = []
            for frame_type in sample_frames_map:
                if frame_type == FRAMES_MAP['P']:
                    sample_frames_encoded.append(P_frames_encoded[P_frame_index])
                    P_frame_index += 1
                elif frame_type == FRAMES_MAP['I']:
                    sample_frames_encoded.append(I_frames[I_frame_index])
                    I_frame_index += 1
                elif frame_type == FRAMES_MAP['<PAD>']:
                    sample_frames_encoded.append(torch.zeros_like(P_frames_encoded[0]))
            frames_encoded.append(torch.stack(sample_frames_encoded, dim=0))
        frames_encoded = self.frames_encoded_dropout(torch.stack(frames_encoded, dim=0))

        frames_encoded = self.frame_type_encoding(frames_encoded, frames_map)
        
        periods_type = torch.cat(
            (
                torch.zeros((frames_encoded.shape[0], self.short_memory_len), dtype=torch.int),
                torch.ones((frames_encoded.shape[0], self.short_memory_len), dtype=torch.int),
                2*torch.ones((frames_encoded.shape[0], self.short_memory_len), dtype=torch.int)
            ),
            dim=1
        ).to(frames_encoded.device) # On time dimension
        if self.two_stage_encoder:
            periods_encoded = torch.cat(
                (
                    self.period_encoder(frames_encoded[:,:self.short_memory_len,:]),
                    self.period_encoder(frames_encoded[:,self.short_memory_len:2*self.short_memory_len,:]),
                    self.period_encoder(frames_encoded[:,2*self.short_memory_len:,:])
                ),
                dim=1
            ) # On time dimension
            periods_encoded = self.period_encoder_dropout(periods_encoded)
            periods_encoded = self.period_type_encoding(periods_encoded, periods_type)
            video_encoded = self.video_encoder(periods_encoded, src_key_padding_mask=temporal_masks)
            video_encoded = self.video_encoder_dropout(video_encoded)
            spotting_head_input = periods_encoded+video_encoded
        else:
            frames_encoded = self.period_type_encoding(frames_encoded, periods_type)
            video_encoded = self.video_encoder(frames_encoded, src_key_padding_mask=temporal_masks)
            video_encoded = self.video_encoder_dropout(video_encoded)
            spotting_head_input = video_encoded

        spotting_head_input = torch.stack(
            (
                torch.mean(spotting_head_input[:,:self.short_memory_len,:], dim=1),
                torch.mean(spotting_head_input[:,self.short_memory_len:2*self.short_memory_len,:], dim=1),
                torch.mean(spotting_head_input[:,2*self.short_memory_len:,:], dim=1)
            ),
            dim=1
        ) # On time dimension
        spotting_logits = self.spotting_head(spotting_head_input)

        if self.umt5_enabled:
            umt5_encoder_outputs = self.umt5_projector(video_encoded)
            if self.use_transcripts_features:
                umt5_encoder_outputs = torch.cat(
                    [
                        umt5_encoder_outputs,
                        self.transcripts_features_dropout(transcripts_features)
                    ],
                    dim=1
                )
        loss = None
        if spotting_labels is not None:
            spotting_loss = self.spotting_loss_fn(spotting_logits, spotting_labels, temporal_masks)
            if self.umt5_enabled:
                umt5_output = self.umt5_model(encoder_outputs=(umt5_encoder_outputs,), attention_mask=captioner_attention_masks, labels=captions_input_ids)
                umt5_logits = umt5_output.logits
                loss = self.spotting_loss_weight*spotting_loss+self.umt5_loss_weight*umt5_output.loss
            else:
                loss = self.spotting_loss_weight*spotting_loss
                umt5_logits = None
        elif self.umt5_enabled:
            umt5_logits = self.umt5_model.generate(attention_mask=captioner_attention_masks,
                                                   encoder_outputs=BaseModelOutput(last_hidden_state=umt5_encoder_outputs),
                                                   **kwargs)
        else:
            umt5_logits = None

        return (
            spotting_logits,
            umt5_logits,
            loss
        )


