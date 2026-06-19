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

from CLIP.clip import create_model
from CLIP.adapter import BCIFL
import random


IFL_PREDICTION_DIR = ""
TEXT_FEATURE_DIR = ""
DATASET_DIR = ""
SAVE_DIR = ""
CHECKPOINT_PATH = ""


def cal_iou(mask, predict):
    intersection = np.logical_and(mask, predict)
    union = np.logical_or(mask, predict)
    return np.sum(intersection, dtype=np.float32) / (np.sum(union, dtype=np.float32) + 1e-8)


def cal_precision(mask, predict):
    TP = ((mask == 1) & (predict == 1))
    FP = ((mask == 1) & (predict == 0))
    return np.sum(TP, dtype=np.float32) / ((np.sum(TP, dtype=np.float32) + np.sum(FP, dtype=np.float32)) + 1e-6)


def cal_recall(mask, predict):
    TP = ((mask == 1) & (predict == 1))
    FN = ((mask == 0) & (predict == 1))
    return np.sum(TP, dtype=np.float32) / ((np.sum(TP, dtype=np.float32) + np.sum(FN, dtype=np.float32)) + 1e-6)


def cal_f1(mask, predict):
    precision = cal_precision(mask, predict)
    recall = cal_recall(mask, predict)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1


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

            return torch.tensor(img_RGB.transpose(2, 0, 1),
                                dtype=torch.float) / 255.0, img_mask.float(), prob, tamper_text_feature, bias_text_feature, rgb_path
        else:
            return torch.tensor(img_RGB.transpose(2, 0, 1), dtype=torch.float) / 255.0, img_mask.float(), rgb_path

    def __len__(self):
        if self.mode == 'train':
            return 2000
        return len(self.files)

    def shuffle(self):
        random.shuffle(self.files)


clip_model = create_model(model_name='ViT-L-14-336', img_size=512, device='cuda', pretrained='openai', require_pretrained=True)
clip_model.eval()

th_model = BCIFL(clip_model=clip_model, features=[6, 12, 18, 24]).cuda()
th_model.eval()
model_param = torch.load(CHECKPOINT_PATH)
th_model.load_state_dict(model_param['th_model_state_dict'])
del model_param

test_dataset = BCIFLDataset(
    data_dir=DATASET_DIR,
    txt_file="",
    mode='test'
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=1,
    drop_last=True,
    num_workers=4
)

f1, iou = [], []

with torch.no_grad():
    for rgb, gt, prob, _ in tqdm(test_dataloader):
        rgb = rgb.cuda()
        gt = gt.numpy()

        _, _, pixel_th, _, _ = th_model(rgb, None, None, mode='train', epoch=39)

        prob = prob.cpu().numpy()

        pixel_th = pixel_th.cpu().numpy()
        pred = (prob > pixel_th).astype('float32')

        f1.append(cal_f1(gt, pred))
        iou.append(cal_iou(gt, pred))

print(
    f'F1: {np.mean(f1):.3f}  '
    f'IOU: {np.mean(iou):.3f}  '
)
