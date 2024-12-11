import os
import glob
import argparse
import PIL.Image
import torch
from transformers import AutoProcessor, CLIPVisionModel
import torchvision.models
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--base_dir',
                    required=True,type=str,
                    help='path to base directory containing data folder')
parser.add_argument('--gop_size',
                    required=False,type=int,
                    help='GOP size',default=-1)
parser.add_argument('--feature_extractor',
                    required=True,type=str,
                    help='feature extractor')
parser.add_argument('--batch_size',
                    required=True,type=int,
                    help='batch size')

args = parser.parse_args()
assert args.feature_extractor in ['resnet', 'clip'], "Feature extractor should be one of 'resnet' or 'clip'"

class FramesDataset(Dataset):
    def __init__(self, base_dir):
        if args.feature_extractor == 'resnet':
            self.processor = transforms.Compose([
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else: # clip
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        if args.gop_size > 0:
            self.frames_path = sorted(glob.glob(f"{base_dir}/data/*/*/*/*_h264_gop{args.gop_size}/*/*I.jpg"))
        else:
            self.frames_path = sorted(glob.glob(f"{base_dir}/data/*/*/*/*_h264/*/*I.jpg"))

    def __len__(self):
        return len(self.frames_path)
    
    def __getitem__(self, index):
        if args.feature_extractor == 'resnet':
            return {
                "frames_path": self.frames_path[index],
                "pixel_values": self.processor(PIL.Image.open(self.frames_path[index]))
            }
        else: # clip
            return {
                "frames_path": self.frames_path[index],
                "pixel_values": self.processor(images=PIL.Image.open(self.frames_path[index]), return_tensors="pt").pixel_values.squeeze()
            }

dataloader = DataLoader(
    FramesDataset(args.base_dir),
    batch_size=args.batch_size,
    num_workers=os.cpu_count(),
    shuffle=False
)

if args.feature_extractor == 'resnet':
    model = torch.nn.Sequential(*list(torchvision.models.resnet152(weights=torchvision.models.ResNet152_Weights.IMAGENET1K_V1).children())[:-1]).to('cuda')
else: # clip
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
model.train(False)

with torch.no_grad():
    for batch in tqdm(dataloader, total=len(dataloader)):
        if args.feature_extractor == 'resnet':
            batch_features = model(batch["pixel_values"].to('cuda')).squeeze()
        else: # clip
            batch_features = model(pixel_values=batch["pixel_values"].to('cuda')).pooler_output.squeeze()
        for sample_index in range(batch_features.shape[0]):
            frame_path = batch["frames_path"][sample_index]
            frame_features_path = f"{frame_path.rsplit('.', 1)[0]}_{args.feature_extractor}.pt"
            torch.save(batch_features[sample_index].detach().cpu(), frame_features_path)
