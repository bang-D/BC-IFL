import os
import torch
import numpy as np
from tqdm import tqdm
from CLIP.tokenizer import tokenize
from CLIP.clip import create_model


def main():
    clip_model = create_model(model_name='ViT-L-14-336', img_size=512, device='cuda', pretrained='openai',
                              require_pretrained=True)
    clip_model.eval()

    text_dir = ''
    feature_dir = ''
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    data_list = open('', 'r').readlines()
    for data in tqdm(data_list):
        name = data.split(',')[0].split('/')[-1].split('.')[0]
        text_file_path = text_dir + f'/{name}.txt'
        text_file = open(text_file_path, 'r')
        text = text_file.read()
        text_file.close()

        text_split = text.split('.')
        if len(text_split) < 3:
            print(name)
            continue
        tamper = text_split[0] + '.'
        bias = ''
        for sentence_idx, sentence in enumerate(text_split):
            if 'The model' in sentence:
                for t in text_split[sentence_idx:]:
                    bias = bias + t + '.'
                break

        final_text_feature = []
        with torch.no_grad():
            text_feature = tokenize(list(tamper)).cuda()
            text_feature = clip_model.encode_text(text_feature)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            text_feature = text_feature.mean(dim=0)
            text_feature = text_feature / text_feature.norm()
            final_text_feature.append(text_feature)
            final_text_feature = torch.stack(final_text_feature, dim=1).cuda()

            final_text_feature = final_text_feature.cpu().numpy()
            np.save(f'{feature_dir}/{name}_tamper.npy', final_text_feature)

        final_text_feature = []
        with torch.no_grad():
            text_feature = tokenize(list(bias)).cuda()
            text_feature = clip_model.encode_text(text_feature)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            text_feature = text_feature.mean(dim=0)
            text_feature = text_feature / text_feature.norm()
            final_text_feature.append(text_feature)
            final_text_feature = torch.stack(final_text_feature, dim=1).cuda()

            final_text_feature = final_text_feature.cpu().numpy()
            np.save(f'{feature_dir}/{name}_bias.npy', final_text_feature)


if __name__ == '__main__':
    main()
