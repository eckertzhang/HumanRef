# HumanRef
Jingbo Zhang, Xiaoyu Li, Qi Zhang, Yanpei Cao, Ying Shan, Jing Liao

| [Project Page](https://eckertzhang.github.io/HumanRef.github.io/) | [Paper](https://arxiv.org/abs/2311.16961) |

This repository contains the official implementation of the paper ["HumanRef: Single Image to 3D Human Generation via Reference-Guided Diffusion"](https://arxiv.org/abs/2311.16961). Note that, this code is forked from [threestudio](https://github.com/threestudio-project/threestudio) for 3D representation and rendering pipeline. 

## Installation
```sh
conda env create -f environment.yml
conda activate humanref

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/huggingface/diffusers.git@ce5504934ac484fca39a1a5434ecfae09eabdf41

git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast && pip install .
cd .. && rm -rf nvdiffrast
```

## Download pre-trained models

### 1. download econ_weights
```sh
mkdir -p Weights/econ_weights && cd Weights
sh ./third_parties/ECON/fetch_data.sh
mv data/* econ_weights/
cd ..
```

### 2. download 'blip2-opt-2.7b' & 'stable-diffusion-v1-5'
```sh
cd Weights
git lfs install
git clone https://huggingface.co/Salesforce/blip2-opt-2.7b
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
cd ..
```

### 3. download weights of LPIPS
```sh
mkdir -p Weights/LPIPS && cd Weights/LPIPS
wget https://download.pytorch.org/models/vgg16-397923af.pth
cd ../..
```

### 4. download weights of SCHP
```sh
mkdir -p Weights/SCHP
```
download [exp-schp-201908261155-lip.pth](https://drive.google.com/file/d/1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH/view?usp=sharing) and put it in 'Weights/SCHP'


## Quickstart
### Preprocessing: Estimate the SMPL-X body mesh using ECON and segment the input image into RGBA format

You need to modify the path setting ('os.environ['WEIGHT_PATH']') in 'third_parties/ECON/run_ECON_smpl.py'.

```sh
python third_parties/ECON/run_ECON_smpl.py --in_dir='./data/image_000355.jpg' --out_dir='./data/Results_ECON'
```

### Optimization

```sh
python run.py --config configs/humanref.yaml --train --gpu 0 image_path='./data/Results_ECON/image_000355/econ/imgs_crop/image_000355_0_rgba.png'
```

## Citation
If you find our code or paper helps, please consider citing:
```
@inproceedings{zhang2024humanref,
  title={Humanref: Single image to 3d human generation via reference-guided diffusion},
  author={Zhang, Jingbo and Li, Xiaoyu and Zhang, Qi and Cao, Yanpei and Shan, Ying and Liao, Jing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1844--1854},
  year={2024}
}
```