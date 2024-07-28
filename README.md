
# AdvDetectGPT: Detecting Adversarial Examples Using Large Vision-Language Models

![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)
![Model Weight License](https://img.shields.io/badge/Model_Weight%20License-CC%20By%20NC%204.0-red.svg)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)


****

## Online Demo

Here, we demonstrate one examples of our online demo. In this example, AdvDetectGPT receives an input image. For the
user's question "Is this an adversarial example or a clean example?", it answers with "This is an adversarial example.".


![AdvDetectGPT_demo](./online_demo.png)
<p align="center">
<img src="./online_demo.png" alt="AdvDetectGPT_demo" style="width: 60%; min-width: 300px; display: block; margin: auto;">
</p>

****

<span id='all_catelogue'/>

## Catalogue
* <a href='#introduction'>1. Introduction</a>
* <a href='#environment'>2. Running AdvDetectGPT Demo</a>
    * <a href='#install_environment'>2.1. Environment Installation</a>
    * <a href='#download_imagebind_model'>2.2. Prepare ImageBind Checkpoint</a>
    * <a href='#download_vicuna_model'>2.3. Prepare Vicuna Checkpoint</a>
    * <a href='#download_advdetectgpt'>2.4. Prepare Delta Weights of AdvDetectGPT</a>
    * <a href='#running_demo'>2.5. Deploying Demo</a>
* <a href='#train_advdetectgpt'>3. Train Your Own AdvDetectGPT</a>
    * <a href='#data_preparation'>3.1. Data Preparation</a>
    * <a href='#training_configurations'>3.2. Training Configurations</a>
    * <a href='#model_training'>3.3. Training AdvDetectGPT</a>
* <a href='#license'>4. Usage and License Notices</a>

****

<span id='introduction'/>

### 1. Introduction <a href='#all_catelogue'>[Back to Top]</a>

<p align="center">
<img src="AdvDetectGPT.png" alt="AdvDetectGPT" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>

AdvDetectGPT is an adversarial detector based on the large vision-language models.
It is composed of an image encoder, a text encoder, and an LLM. In this setup, both
the image encoder and the text encoder are frozen, while the LLM is fine-tuned using the LoRA technique.
AdvDetectGPT can learn to identify adversarial examples directly from clean and adversarial instances,
independent of the victim model's outputs or internal features.


****

<span id='environment'/>

### 2. Running AdvDetectGPT Demo <a href='#all_catelogue'>[Back to Top]</a>

<span id='install_environment'/>

#### 2.1. Environment Installation
To install the required packages, please run:
```
pip install -r requirements.txt
```

Then install the PyTorch with the correct cuda version, for example:
```
pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch/
```

<span id='download_imagebind_model'/>

#### 2.2. Prepare ImageBind Checkpoint
You can download the pre-trained ImageBind model using [this link](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth). After downloading, put the downloaded file (imagebind_huge.pth) in [./pretrained_ckpt/imagebind_ckpt/](./pretrained_ckpt/imagebind_ckpt/) directory. 

<span id='download_vicuna_model'/>

#### 2.3. Prepare Vicuna Checkpoint
To prepare the pre-trained Vicuna model, please follow the instructions provided [here](./pretrained_ckpt/README.md#1-prepare-vicuna-checkpoint).


<span id='download_advdetectgpt'/>

#### 2.4. Prepare Delta Weights of AdvDetectGPT
The delta weights of AdvDetectGPT have already been placed in the directory [./code/ckpt/advgpt_imagenet_7b_peft/](./code/ckpt/advgpt_imagenet_7b_peft/)

<span id='deploying_demo'/>

#### 2.5. Deploying Demo
Upon completion of previous steps, you can run the demo locally as
```bash
cd ./code/
CUDA_VISIBLE_DEVICES=0 python web_demo.py
```

If you running into `sample_rate` problem, please git install `pytorchvideo` from the source as
```yaml
git clone https://github.com/facebookresearch/pytorchvideo
cd pytorchvideo
pip install --editable ./
```

****

<span id='train_advdetectgpt'/>

### 3. Train Your Own AdvDetectGPT <a href='#all_catelogue'>[Back to Top]</a>

Before training the model, making sure the environment is properly installed and the checkpoints of ImageBind and
Vicuna are downloaded.

<span id='data_preparation'/>

#### 3.1. Data Preparation

Please prepare training data on your own and place it in the direcotry  [./data/](./data/).

The directory should look like:

    .
    └── ./data/ 
         ├── advgpt_visual_instruction_imagenet_data.json
         └── /images/
             ├── 00000004.jpg
             ├── 00000012.jpg
             └── ...


<span id='model_training'/>

#### 3.3. Training AdvDetectGPT
 
To train AdvDetectGPT, please run the following commands:
```yaml
cd ./code/scripts/
chmod +x train_advgpt.sh
cd ..
./scripts/train_advgpt.sh
```

The key arguments of the training script are as follows:
* `--data_path`: The data path for the json file `advgpt_visual_instruction_imagenet_data.json`.
* `--image_root_path`: The root path for the training images.
* `--imagebind_ckpt_path`: The path where saves the ImageBind checkpoint `imagebind_huge.pth`.
* `--vicuna_ckpt_path`: The path that saves the pre-trained Vicuna checkpoints.
* `--max_tgt_len`: The maximum sequence length of training instances.
* `--save_path`: The directory which saves the trained delta weights. This directory will be automatically created.

Note that the epoch number can be set in the `epochs` argument at [./code/config/openllama_peft.yaml](./code/config/openllama_peft.yaml) file. 
The `train_micro_batch_size_per_gpu` and `gradient_accumulation_steps` arguments in
[./code/dsconfig/openllama_peft_stage_1.json](./code/dsconfig/openllama_peft_stage_1.json) should be set as `2` and `4` for 7B model, 
and set as `1` and `8` for 13B model.

****

<span id='license'/>

### 4. Usage and License Notices

AdvDetectGPT is intended and licensed for research use only. The dataset is CC BY NC 4.0 (allowing only non-commercial use) 
and models trained using the dataset should not be used outside of research purposes. The delta weights are also CC BY 
NC 4.0 (allowing only non-commercial use).
