import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

qwen_weight_path = ""
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    qwen_weight_path, torch_dtype="auto", device_map="auto"
)

# default processer
processor = AutoProcessor.from_pretrained(qwen_weight_path)

txt_file = '/data0/denghaoyi/BestTh/Mesorch/train_data_53/train_data.txt'

save_dir = ""
img_dir = ""
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data_list = open(txt_file, 'r').readlines()
for data_id, data in enumerate(tqdm(data_list)):
    name = data.split('/')[-1].split('.')[0]

    img_path = img_dir + f'/{name}.png'
    record_txt = open(f'{save_dir}/{name}.txt', 'w')

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {
                    "type": "text",
                    "text": "You are an image forgery identification assistant. "
                            "This is a tampered image. The red box marks the tampered region, and the green box is "
                            "the judgment result of a Image Forgery Localization model. Please describe the tampered region"
                            "and the ability of the model to detect the tampered region. "
                            "Please answer 'The tampered content is XXX which locates in left/right/center, and it is large/small/medium. "
                            "The model has high/low accuracy in detect the tampered region and has high/low accuracy in identify the pristine region.'"
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    output_text = output_text[0].replace('Problem 1:', '').replace('Problem 2:', '').replace('Problem 3:', '').replace('\n', '')
    record_txt.write(output_text)
    record_txt.close()
