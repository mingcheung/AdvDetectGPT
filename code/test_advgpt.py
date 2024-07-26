import os
from model.openllama import OpenLLAMAPEFTModel
import pandas as pd
import torch

# init the model
args = {
    'model': 'openllama_peft',
    'imagebind_ckpt_path': '../pretrained_ckpt/imagebind_ckpt',
    'vicuna_ckpt_path': '../pretrained_ckpt/vicuna_ckpt/7b_v0',
    'delta_ckpt_path': './ckpt/advgpt_imagenet_7b_peft/pytorch_model.pt',
    'stage': 2,
    'max_tgt_len': 64,
    'lora_r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}
model = OpenLLAMAPEFTModel(**args)
delta_ckpt = torch.load(args['delta_ckpt_path'], map_location=torch.device('cpu'))
model.load_state_dict(delta_ckpt, strict=False)
model = model.eval().half().cuda()
print(f'[!] init the 7b model over ...')


df = pd.read_csv('path_to_imagenet_compatible_dataset/imagenet_compatible_dataset_labels.csv')
file_names = list(df['ImageId'])
file_names = [image_id+'.png' for image_id in file_names]

dirs = ['clean_images', 'adv_images_eps_8', 'adv_images_eps_16', 'adv_images_eps_32']

for dir in dirs:
    os.makedirs('./testing_images/detection_results/' + dir, exist_ok=True)
    sub_dirs = os.listdir('./testing_images/' + dir)
    for sub_dir in sub_dirs:
        save_path = os.path.join('./testing_images/detection_results/'+dir, sub_dir+'_detection_results.txt')
        fw = open(save_path, 'w')
        for f_name in file_names:
            input = 'Is this an adversarial example or a clean example?'
            prompt_text = f'{input}'
            image_path = os.path.join('./testing_images', dir, sub_dir, f_name)
            response = model.generate({
                'prompt': prompt_text,
                'image_paths': [image_path] if image_path else [],
                'audio_paths': [],
                'video_paths': [],
                'thermal_paths': [],
                'top_p': 0.01,
                'temperature': 1.0,
                'max_tgt_len': 64,
                'modality_embeds': []
            })
            fw.write(f_name + ',' + response + '\n')
        fw.close()
