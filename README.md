# Reducio-VAE
Welcome to the official repository for **Reducio Varitional Autoencoder (Reducio-VAE)**! [Reducio-VAE](https://arxiv.org/abs/) is a model for encoding videos into an extremely small latent space. It is part of the Reducio-DiT, which is a highly efficient video generation method. Reducio-VAE encodes a 16-frame video clip to $T/4*H/32*W/32$ latent space based on a content image prior, which enables 4096x compression rate on the videos. More details can be found in the paper.

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/) [![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/microsoft/Reducio-VAE)  


## WHAT CAN REDUCIO-VAE DO
Reducio-VAE was developed to enable high compression ratio on videos, supporting efficient video generation. Existing 3D VAEs are generally extended from 2D VAE, which is designed for image generation and has large redundancy when handling video. Compared to 2D VAE, Reducio-VAE achieved 64x high compression ratio.

A detailed discussion of Reducio-VAE, including how it was developed and tested, can be found in our [paper](https://arxiv.org/abs/).

## INTENDED USES
Reducio-VAE is best suited for supporting training your own video diffusion model for research purpose.

## OUT-OF-SCOPE USES
Reducio-VAE is not well suited for processing long videos. It currently can only handle 1-second video clips.

We do not recommend using Reducio-VAE in commercial or real-world applications without further testing and development. It is being released for research purposes.

Reducio-VAE was not designed or evaluated for all possible downstream purposes. Developers should consider its inherent limitations (more below) as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness concerns specific to each intended downstream use.

We do not recommend using Reducio-VAE in the context of high-risk decision making (e.g. in law enforcement, legal, finance, or healthcare).


## Installation
```
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade diffusers[torch]==0.29.0
pip3 install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==2.0.1 psutil omegaconf==2.3.0
pip install opencv-python scikit-image timm==0.9.16 tokenizers==0.15.2
pip install scipy==1.9.1 tensorboardx==2.6 termcolor==2.4.0 
pip install pandas==2.0.3 transformers==4.33.3 
pip install einops datasets numpy==1.24.4 accelerate==0.32.1
pip install av==11.0.0 basicsr==1.4.2 decord loralib
pip install open_clip_torch kornia optimum bs4 wandb 
pip install sentencepiece~=0.1.99 ftfy beautifulsoup4
```


## TRAINING

### Training Method
Training data was prepared by self-collecting. We recommend using Pexels or other high-quality videos.

We directly trained Reducio-VAE from scratch on $256*256$ resolution with 16 FPS. No further fine-tuning on $512*512$ or $1024*1024$. To use it on higher resolution, we split videos into tiles in spatial dimensions.

### Training Scripts
```
python -m torch.distributed.launch --nproc_per_node=${GPU_PER_NODE_COUNT} \
--node_rank=${NODE_RANK} \
--nnodes=${NODE_COUNT} \
--master_addr=${MASTER_ADDR} \
--master_port=${MASTER_PORT} \
--use_env main.py -b ${config} -t -r ${output_dir} -k ${wandb_key}
```

## Model Zoo
| name |  $f_t$ | $f_s$ |  checkpoint |
|:---:|:---:|:---:|:---:|
 Reducio-VAE | 2 | 32 | |
| Reducio-VAE | 4 | 32 | |

## EVALUATION

Reducio-VAE performed best on high-quality videos, but worse on low-quality videos like UCF-101. The reason is that the visual quality of UCF-101 is low, where the prior content image is blurring in many cases.

Metrics on 1K Pexels validation set and UCF-101: 

|Method|Downsample Factor|$\|z\|$|PSNR |SSIM |LPIPS |rFVD (Pexels)|rFVD (UCF-101)|
|---------|---------------------|------------------|------------|--------------------|--------------|----------------|------------|
|SD2.1-VAE|$1\times8\times8$|4|29.23|0.82|0.09|25.96|21.00| 
|SDXL-VAE|$1\times8\times8$|16|30.54|0.85|0.08|19.87|23.68|
|OmniTokenizer|$4\times8\times8$|8|27.11|0.89|0.07|23.88|30.52|
|OpenSora-1.2|$4\times8\times8$|16|30.72|0.85|0.11|60.88|67.52|
|Cosmos Tokenizer|$8\times8\times8$|16|30.84|0.74|0.12|29.44|22.06|
|Cosmos Tokenizer|$8\times16\times16$|16|28.14|0.65|0.18|77.87|119.37|
|Reducio-VAE|$4\times32\times32$|16|35.88|0.94|0.05|17.88|65.17|


## LIMITATIONS
Reducio-VAE was developed for research and experimental purposes. Further testing and validation are needed before considering its application in commercial or real-world scenarios.

Currently, Reducio-VAE can only encode 16-frame video clips. Longer length is not supported.

## LICENSE
The code and model are licensed under the MIT license.

## Acknowledgements

Our code is built on top of [Stable Diffusion](https://github.com/Stability-AI/). We would like to thank the SD team for their foundational work.

## Citation

If you use our work, please cite:

```
@article{tian2024reducio,
  title={REDUCIO! Generating 1024*1024 Video within 16 Seconds using Extremely Compressed Motion Latents},
  author={Tian, Rui and Dai, Qi and Bao, Jianmin and Qiu, Kai and Yang, Yifan and Luo, Chong and Wu, Zuxuan and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:},
  year={2024}
}
