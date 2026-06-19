import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import sys, os
import numpy as np
from tqdm import tqdm
import torch

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn

from CLIP.clip import create_model
from CLIP.adapter import BCIFL
import random


IFL_PREDICTION_DIR = ""
TEXT_FEATURE_DIR = ""
DATASET_DIR = ""
SAVE_DIR = ""


def set_seed(seed=98):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BCIFLDataset(Dataset):
    def __init__(self, data_dir, txt_file, mode):
        f = open(txt_file, 'r')
        self.files = f.readlines()
        f.close()

        self.data_dir = data_dir
        self.mode = mode

        transforms_mask_list = [transforms.ToPILImage(), transforms.ToTensor()]
        self.transform_mask = transforms.Compose(transforms_mask_list)

    def __getitem__(self, idx):
        rgb_path = self.data_dir + self.files[idx].split(',')[0]
        name = rgb_path.split('/')[-1].split('.')[0]
        mask_path = self.data_dir + self.files[idx].strip().split(',')[1]

        img_RGB = cv2.cvtColor(cv2.imread(rgb_path, 1), cv2.COLOR_BGR2RGB)
        img_mask = cv2.imread(mask_path, 0)

        prob = cv2.imread(f'{IFL_PREDICTION_DIR}/{name}.png', 0)

        img_RGB = cv2.resize(img_RGB, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_mask = cv2.resize(img_mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        img_mask = self.transform_mask(img_mask)
        prob = self.transform_mask(prob)

        if self.mode == 'train':
            tamper_text_feature = np.load(f'{TEXT_FEATURE_DIR}/{name}_tamper.npy')
            bias_text_feature = np.load(f'{TEXT_FEATURE_DIR}/{name}_bias.npy')

            return torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 255.0, img_mask.float(), prob, tamper_text_feature, bias_text_feature, rgb_path
        else:
            return torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 255.0, img_mask.float(), rgb_path

    def __len__(self):
        if self.mode == 'train':
            return 2000
        return len(self.files)

    def shuffle(self):
        random.shuffle(self.files)


class TextAdapter(nn.Module):
    def __init__(self):
        super(TextAdapter, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 768)

    def forward(self, text_feature):
        text_feature = text_feature.permute(0, 2, 1)
        return self.fc2(self.fc1(text_feature)).permute(0, 2, 1)


def main():
    set_seed()

    clip_model = create_model(model_name='ViT-L-14-336', img_size=512, device='cuda', pretrained='openai', require_pretrained=True)
    clip_model.eval()

    th_model = BCIFL(clip_model=clip_model, features=[6, 12, 18, 24]).cuda()
    th_model.eval()

    tamper_text_adapter = TextAdapter().cuda().train()
    bias_text_adapter = TextAdapter().cuda().train()

    train_dataset = BCIFLDataset(
        data_dir=DATASET_DIR,
        txt_file="",
        mode='train'
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=1,
        drop_last=True,
        num_workers=0
    )

    optimizer = torch.optim.AdamW(
        list(th_model.tamper_adapters.parameters()) + list(th_model.tamper_ffn.parameters()) + list(
            th_model.bias_adapters.parameters()) + list(th_model.bias_ffn.parameters()) + list(
            th_model.dbcp.parameters()) + list(
            tamper_text_adapter.parameters()) + list(bias_text_adapter.parameters()), lr=1e-4, weight_decay=0.0,
        betas=(0.9, 0.999))

    for epoch in range(40):
        mean_loss, mean_bias_loss, mean_tamper_loss, mean_th_loss = [], [], [], []
        for batch_idx, (rgb, gt, prob, tamper_text_feature, bias_text_feature, name) in enumerate(tqdm(train_dataloader)):
            rgb, gt, prob, tamper_text_feature, bias_text_feature = rgb.cuda(), gt.cuda(), prob.cuda(), tamper_text_feature.cuda(), bias_text_feature.cuda()

            tamper_text_feature = tamper_text_adapter(tamper_text_feature)
            bias_text_feature = bias_text_adapter(bias_text_feature)

            bias_prediction, tamper_prediction, pixel_th, bias_cls_token, tamper_cls_token = th_model(rgb, None, None, mode='train', epoch=epoch)

            loss_bias = nn.MSELoss()(bias_prediction, torch.abs(prob - gt))
            loss_tamper = nn.BCELoss()(tamper_prediction, gt)

            bias_bias = bias_cls_token / bias_cls_token.norm(dim=-1, keepdim=True) @ bias_text_feature.squeeze(0)
            bias_tamper = bias_cls_token / bias_cls_token.norm(dim=-1, keepdim=True) @ tamper_text_feature.squeeze(0)
            tamper_bias = tamper_cls_token / tamper_cls_token.norm(dim=-1, keepdim=True) @ bias_text_feature.squeeze(0)
            tamper_tamper = tamper_cls_token / tamper_cls_token.norm(dim=-1, keepdim=True) @ tamper_text_feature.squeeze(0)

            loss_bias += torch.nn.functional.cross_entropy(
                torch.tensor([bias_bias, bias_tamper], dtype=rgb.dtype, device=rgb.device).unsqueeze(0),
                torch.tensor([0], device=rgb.device))
            loss_tamper += torch.nn.functional.cross_entropy(
                torch.tensor([tamper_bias, tamper_tamper], dtype=rgb.dtype, device=rgb.device).unsqueeze(0),
                torch.tensor([1], device=rgb.device))

            loss_th = nn.MSELoss()(prob - pixel_th, (2 * gt) - 0.5)

            loss = loss_bias + loss_tamper + loss_th

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss.append(loss.item())
            mean_bias_loss.append(loss_bias.item())
            mean_tamper_loss.append(loss_tamper.item())
            mean_th_loss.append(loss_th.item())

        print(
            f'==> Train:  Epoch: {epoch}  '
            f'Total Loss: {np.mean(mean_loss):.4f}  '
            f'Bias Loss: {np.mean(mean_bias_loss):.4f}  '
            f'Tamper Loss: {np.mean(mean_tamper_loss):.4f}  '
            f'Th Loss: {np.mean(mean_th_loss):.4f}  ')

        train_dataset.shuffle()

        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    'th_model_state_dict': th_model.state_dict(),
                }, f'{SAVE_DIR}/checkpoint/Localization-epoch={epoch}.pt'
            )
