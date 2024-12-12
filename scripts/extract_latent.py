import argparse
import tqdm
import os
import torch
import diffusers
import pandas as pd
import decord
import numpy as np
import einops
import safetensors
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision
import torch.nn
from omegaconf import OmegaConf
import imageio
import PIL.Image as Image
from pathlib import Path
import sys
import open_clip
import kornia
from bisect import bisect_left
from transformers import T5EncoderModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from bs4 import BeautifulSoup
import html
import ftfy
import re
import urllib.parse as ul
import random

decord.bridge.set_bridge("torch")
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
from ldm.util import instantiate_from_config


ASPECT_RATIO_1024 = {
    '0.25': [512., 2048.], '0.26': [512., 1984.], '0.27': [512., 1920.], '0.28': [512., 1856.],
    '0.32': [576., 1792.], '0.33': [576., 1728.], '0.35': [576., 1664.], '0.4':  [640., 1600.],
    '0.42':  [640., 1536.], '0.48': [704., 1472.], '0.5': [704., 1408.], '0.52': [704., 1344.],
    '0.57': [768., 1344.], '0.6': [768., 1280.], '0.68': [832., 1216.], '0.72': [832., 1152.],
    '0.78': [896., 1152.], '0.82': [896., 1088.], '0.88': [960., 1088.], '0.94': [960., 1024.],
    '1.0':  [1024., 1024.], '1.07': [1024.,  960.], '1.13': [1088.,  960.], '1.21': [1088.,  896.],
    '1.29': [1152.,  896.], '1.38': [1152.,  832.], '1.46': [1216.,  832.], '1.67': [1280.,  768.],
    '1.75': [1344.,  768.], '2.0':  [1408.,  704.], '2.09':  [1472.,  704.], '2.4':  [1536.,  640.],
    '2.5':  [1600.,  640.], '2.89':  [1664.,  576.], '3.0':  [1728.,  576.], '3.11':  [1792.,  576.],
    '3.62':  [1856.,  512.], '3.75':  [1920.,  512.], '3.88':  [1984.,  512.], '4.0':  [2048.,  512.],}

ASPECT_RATIO_512 = {
     '0.25': [256.0, 1024.0], '0.26': [256.0, 992.0], '0.27': [256.0, 960.0], '0.28': [256.0, 928.0],
     '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
     '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
     '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
     '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
     '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
     '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
     '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
     '2.5': [800.0, 320.0], '2.89': [832.0, 288.0], '3.0': [864.0, 288.0], '3.11': [896.0, 288.0],
     '3.62': [928.0, 256.0], '3.75': [960.0, 256.0], '3.88': [992.0, 256.0], '4.0': [1024.0, 256.0]}


# ASPECT_RATIO_256 ={
#     '0.25': [128.0, 512.0], '0.27': [128.0, 480.0], '0.42': [160.0, 384.0], 
#     '0.6': [192.0, 320.0], '0.78': [224.0, 288.0], '1.0': [256.0, 256.0], 
#     '1.29': [288.0, 224.0], '1.67': [320.0, 192.0], '2.4': [384.0, 160.0], 
#     '3.75': [480.0, 128.0], '4.0': [512.0, 128.0]}

ASPECT_RATIO_256={
     '0.25': [128.0, 512.0], '0.26': [128.0, 496.0], '0.27': [128.0, 480.0], '0.28': [128.0, 464.0],
     '0.32': [144.0, 448.0], '0.33': [144.0, 432.0], '0.35': [144.0, 416.0], '0.4': [160.0, 400.0],
     '0.42': [160.0, 384.0], '0.48': [176.0, 368.0], '0.5': [176.0, 352.0], '0.52': [176.0, 336.0],
     '0.57': [192.0, 336.0], '0.6': [192.0, 320.0], '0.68': [208.0, 304.0], '0.72': [208.0, 288.0],
     '0.78': [224.0, 288.0], '0.82': [224.0, 272.0], '0.88': [240.0, 272.0], '0.94': [240.0, 256.0],
     '1.0': [256.0, 256.0], '1.07': [256.0, 240.0], '1.13': [272.0, 240.0], '1.21': [272.0, 224.0],
     '1.29': [288.0, 224.0], '1.38': [288.0, 208.0], '1.46': [304.0, 208.0], '1.67': [320.0, 192.0],
     '1.75': [336.0, 192.0], '2.0': [352.0, 176.0], '2.09': [368.0, 176.0], '2.4': [384.0, 160.0],
     '2.5': [400.0, 160.0], '2.89': [416.0, 144.0], '3.0': [432.0, 144.0], '3.11': [448.0, 144.0],
     '3.62': [464.0, 128.0], '3.75': [480.0, 128.0], '3.88': [496.0, 128.0], '4.0': [512.0, 128.0]}


def save_tensor_to_video(videos, p_save, save_type='gif', duration=None, best_quality=False):
    """
    save video tensors as gif file
    """
    if isinstance(videos, torch.Tensor):
        videos = videos.unbind(0)
        videos = [video.unsqueeze(0) for video in videos]
    for idx, video in enumerate(videos):
        video = (video.cpu() + 1.0) / 2.0 
        frames = [video[:, :, i] for i in range(video.shape[2])]
        frames = [torchvision.utils.make_grid(each, nrow=8) for each in frames]
        frames = [einops.rearrange(each, "c h w -> 1 c h w") for each in frames]
        frames = torch.clamp(torch.cat(frames, dim=0), min=0.0, max=1.0)
        frames = (frames.numpy() * 255).astype(np.uint8)
        image_list = []
        p_save = p_save + f'_{idx:03}'
        for frame in frames:
            image = frame.transpose(1, 2, 0)
            image_list.append(image)
        if save_type == 'gif':
            if duration:
                imageio.mimsave(p_save + '.gif', image_list, format="GIF", duration=duration, loop=0)
            else:
                imageio.mimsave(p_save + '.gif', image_list, format="GIF", loop=0)
        elif save_type == 'mp4':
            if best_quality:
                with imageio.get_writer(p_save + '.mp4', fps=fps, quality=10, codec='libx264', pixelformat='yuv444p') as writer:
                    for frame in x:
                        writer.append_data(image_list)
            else:
                with imageio.get_writer(p_save + '.mp4', fps=fps) as writer:
                    for frame in x:
                        writer.append_data(image_list)
        else:
            os.makedirs(p_save, exist_ok=True)
            for idx, frame in enumerate(image_list):
                p_save_tmp = os.path.join(p_save, f"{idx:05}.png")
                Image.fromarray(frame).save(p_save_tmp)


# use t5 encode
class T5Embedder:
    available_models = ['t5-v1_1-xxl']
    bad_punct_regex = re.compile(r'['+'#®•©™&@·º½¾¿¡§~'+'\)'+'\('+'\]'+'\['+'\}'+'\{'+'\|'+'\\'+'\/'+'\*' + r']{1,}')  # noqa
    def __init__(self, device, dir_or_name='t5-v1_1-xxl', *, local_cache=False, cache_dir=None, hf_token=None, use_text_preprocessing=True,
                 t5_model_kwargs=None, torch_dtype=None, use_offload_folder=None, model_max_length=120):
        self.device = device
        self.torch_dtype = torch_dtype or torch.bfloat16
        if t5_model_kwargs is None:
            t5_model_kwargs = {'low_cpu_mem_usage': True, 'torch_dtype': self.torch_dtype}
            if use_offload_folder is not None:
                t5_model_kwargs['offload_folder'] = use_offload_folder
                t5_model_kwargs['device_map'] = {
                    'shared': self.device,
                    'encoder.embed_tokens': self.device,
                    'encoder.block.0': self.device,
                    'encoder.block.1': self.device,
                    'encoder.block.2': self.device,
                    'encoder.block.3': self.device,
                    'encoder.block.4': self.device,
                    'encoder.block.5': self.device,
                    'encoder.block.6': self.device,
                    'encoder.block.7': self.device,
                    'encoder.block.8': self.device,
                    'encoder.block.9': self.device,
                    'encoder.block.10': self.device,
                    'encoder.block.11': self.device,
                    'encoder.block.12': 'disk',
                    'encoder.block.13': 'disk',
                    'encoder.block.14': 'disk',
                    'encoder.block.15': 'disk',
                    'encoder.block.16': 'disk',
                    'encoder.block.17': 'disk',
                    'encoder.block.18': 'disk',
                    'encoder.block.19': 'disk',
                    'encoder.block.20': 'disk',
                    'encoder.block.21': 'disk',
                    'encoder.block.22': 'disk',
                    'encoder.block.23': 'disk',
                    'encoder.final_layer_norm': 'disk',
                    'encoder.dropout': 'disk',
                }
            else:
                t5_model_kwargs['device_map'] = {'shared': self.device, 'encoder': self.device}

        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token
        self.cache_dir = cache_dir or os.path.expanduser('~/.cache/IF_')
        self.dir_or_name = dir_or_name
        tokenizer_path, path = dir_or_name, dir_or_name
        if local_cache:
            cache_dir = os.path.join(self.cache_dir, dir_or_name)
            tokenizer_path, path = cache_dir, cache_dir
        elif dir_or_name in self.available_models:
            cache_dir = os.path.join(self.cache_dir, dir_or_name)
            for filename in [
                'config.json', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json',
                'pytorch_model.bin.index.json', 'pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin'
            ]:
                hf_hub_download(repo_id=f'DeepFloyd/{dir_or_name}', filename=filename, cache_dir=cache_dir,
                                force_filename=filename, token=self.hf_token)
            tokenizer_path, path = cache_dir, cache_dir
        else:
            cache_dir = os.path.join(self.cache_dir, 't5-v1_1-xxl')
            for filename in [
                'config.json', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json',
            ]:
                hf_hub_download(repo_id='DeepFloyd/t5-v1_1-xxl', filename=filename, cache_dir=cache_dir,
                                force_filename=filename, token=self.hf_token)
            tokenizer_path = cache_dir

        print(tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = T5EncoderModel.from_pretrained(path, **t5_model_kwargs).eval()
        self.model_max_length = model_max_length
        self.caption_channels = self.model.config.d_model

    def get_text_embeddings(self, texts):
        texts = [self.text_preprocessing(text) for text in texts]
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        text_tokens_and_mask['input_ids'] = text_tokens_and_mask['input_ids'].to(self.device)
        text_tokens_and_mask['attention_mask'] = text_tokens_and_mask['attention_mask'].to(self.device)
        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=text_tokens_and_mask['input_ids'],
                attention_mask=text_tokens_and_mask['attention_mask'],
            )['last_hidden_state'].detach()
        
        return text_encoder_embs, text_tokens_and_mask['attention_mask'].to(self.device)

    def text_preprocessing(self, text):
        if self.use_text_preprocessing:
            text = self.clean_caption(text)
            text = self.clean_caption(text)
            return text
        else:
            return text.lower().strip()

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub('<person>', 'person', caption)
        # urls:
        caption = re.sub(
            r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
            '', caption)  # regex for urls
        caption = re.sub(
            r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
            '', caption)  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features='html.parser').text

        # @<nickname>
        caption = re.sub(r'@[\w\d]+\b', '', caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
        caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
        caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
        caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
        caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
        caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
        caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',  # noqa
            '-', caption)

        # кавычки к одному стандарту
        caption = re.sub(r'[`´«»“”¨]', '"', caption)
        caption = re.sub(r'[‘’]', "'", caption)

        # &quot;
        caption = re.sub(r'&quot;?', '', caption)
        # &amp
        caption = re.sub(r'&amp', '', caption)

        # ip adresses:
        caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)

        # article ids:
        caption = re.sub(r'\d:\d\d\s+$', '', caption)

        # \n
        caption = re.sub(r'\\n', ' ', caption)

        # "#123"
        caption = re.sub(r'#\d{1,3}\b', '', caption)
        # "#12345.."
        caption = re.sub(r'#\d{5,}\b', '', caption)
        # "123456.."
        caption = re.sub(r'\b\d{6,}\b', '', caption)
        # filenames:
        caption = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)

        #
        caption = re.sub(r'[\"\']{2,}', r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r'[\.]{2,}', r' ', caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r' ', caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r'\s+\.\s+', r' ', caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r'(?:\-|\_)')
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, ' ', caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)  # jc6640
        caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)  # jc6640vc
        caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)  # 6640vc231

        caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
        caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
        caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
        caption = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
        caption = re.sub(r'\bpage\s+\d+\b', '', caption)

        caption = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)  # j2d1a2a...

        caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)

        caption = re.sub(r'\b\s+\:\s+', r': ', caption)
        caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
        caption = re.sub(r'\s+', ' ', caption)

        caption.strip()

        caption = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
        caption = re.sub(r'^[\'\_,\-\:;]', r'', caption)
        caption = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption)
        caption = re.sub(r'^\.\S+$', '', caption)

        return caption.strip()

def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f"Loaded model config from [{config_path}]")
    return model

class LatentExtractor:
    def __init__(
            self,
            enable_2d=True,
            enable_openclip=False,
            enable_t5=False,
            enable_center=True,
            size=(512, 512),
            aspect_ratio_key=None,
            ar=None,
            save_precision='half',
            autocast_dtype='fp32',
            load_nframes=16,
            vae2d_type='stabilityai/sd-vae-ft-ema',
            *args,
            **kwargs):
        self.h, self.w = size
        self.save_precision = save_precision
        self.autocast_dtype = self.build_autocast_dtype(autocast_dtype)
        self.enable_2d= enable_2d
        self.enable_t5= enable_t5
        self.enable_center = enable_center
        self.enable_openclip = enable_openclip
        self.load_nframes=load_nframes
        if ar is None:
            self.aspect_ratio_key = {'1.0': size} if aspect_ratio_key is None else aspect_ratio_key
            self.aspect_ratio = [] if aspect_ratio_key is None else [key for key, val in aspect_ratio_key.items()]
            self.aspect_ratio_raw = [] if aspect_ratio_key is None else [float(val[0]) / float(val[1]) for key, val in aspect_ratio_key.items()]
            self.aspect_ratio_free = len(self.aspect_ratio) > 0
        else:
            assert ar in aspect_ratio_key
            self.aspect_ratio = [ar]
            self.aspect_ratio_free = False
            self.h, self.w = aspect_ratio_key[ar]
        print(self.aspect_ratio)
        
        vae3d_config, vae3d_p_ckpt = kwargs.get('vae3d_config', None), kwargs.get('vae3d_p_ckpt', None)
        if enable_openclip:
            openclip_model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", device=torch.device("cuda"), pretrained="laion2b_s32b_b79k")
            self.openclip_model = openclip_model
            self.openclip_model.visual.output_tokens = True
            self.mean =  torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to('cuda')
            self.std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to('cuda')
        
        if enable_2d:
            self.vae2d = self.build_2dvae(vae2d_type)
        if enable_t5:
            self.text_embedder = T5Embedder(device="cuda", torch_dtype=torch.float)
            
        self.vae3d = self.build_3dvae(vae3d_config, vae3d_p_ckpt)
        self.vae3d_nframes = kwargs.get('vae3d_nframes', 16)
        self.tsfm = prepare_tfsm(self.h, self.w, len(self.aspect_ratio) > 0)

    def preprocess_openclip(self, x):
        x = kornia.geometry.resize(x, (224, 224), interpolation="bicubic", align_corners=True, antialias=True )
        x = (x + 1.0) / 2.0 
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x
    
    def build_autocast_dtype(self, autocast_dtype):
        if autocast_dtype == 'bf16':
            autocast_dtype = torch.bfloat16
        elif autocast_dtype == 'fp16':
            autocast_dtype = torch.float16
        elif autocast_dtype == 'fp32':
            autocast_dtype = torch.float32
        else:
            raise ValueError('Unknown autocast_dtype: {}'.format(autocast_dtype))
        return autocast_dtype

    def build_2dvae(self, vae2d_type):
        print(f"vae2d_type: {vae2d_type}")
        model = diffusers.models.AutoencoderKL.from_pretrained(vae2d_type)
        del model.decoder
        return model.to("cuda").eval()
    
    def build_3dvae(self, vae3d_config, vae3d_p_ckpt):
        model = create_model(vae3d_config)
        if vae3d_p_ckpt is not None:
            ckpt = torch.load(vae3d_p_ckpt, map_location="cpu")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            msg = model.load_state_dict(ckpt, strict=False)
            print(f'Load Reducio-VAE from {vae3d_p_ckpt}:', msg)
        return model.to("cuda").eval()

    def preprocess_vae(self, x):
        T, C, H, W = x.shape
        if self.aspect_ratio_free:
            ar = float(H) / float(W)
            best_ar = self.aspect_ratio[get_best_aspect_ratio(self.aspect_ratio_raw, ar)]
            new_H, new_W = self.aspect_ratio_key[best_ar]
            if ar <= 1.0:
                x = transforms.Resize((int(new_H), int(new_H / ar)), antialias=True)(x)
            else:
                x = transforms.Resize((int(new_W * ar), int(new_W)), antialias=True)(x)
            x = transforms.CenterCrop((int(new_H), int(new_W)))(x)
            return self.tsfm(x), best_ar
        elif len(self.aspect_ratio) > 0:
            new_H, new_W = self.h, self.w
            ar = float(H) / float(W)
            best_ar = float(new_H) / float(new_W)
            if best_ar >= ar:
                x = transforms.Resize((int(new_H), int(new_H / ar)), antialias=True)(x)
            else:
                x = transforms.Resize((int(new_W * ar), int(new_W)), antialias=True)(x)
            x = transforms.CenterCrop((int(new_H), int(new_W)))(x)
            return self.tsfm(x), best_ar
        else:
            return self.tsfm(x), 1.0
            
    def encode_video(self, x, return_cframe=False):
        nframes = self.vae3d_nframes
        with torch.no_grad():
          with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
            B, C, T, H, W = x.shape
            assert B == 1, 'Batch size must be 1'
            assert T % nframes == 0
            z_list = []
            z2d_list = []
            z_clip_list = []
            cframe_list = []
            for start_idx in range(0, T, nframes):
                end_idx = start_idx + nframes
                center_idx = (nframes - 1) // 2 
                x_batch = x[:, :, start_idx:end_idx, :, :]
                posterior = self.vae3d.encode_3d(x_batch.cuda(), use_tiling=True)
                z = posterior.sample()
                z = torch.nn.functional.layer_norm(z, z.shape[1:])
                z_list.append(z.half() if self.save_precision == 'half' else z.float())
                if self.enable_2d:
                    posterior =  self.vae2d.encode(x_batch[:, :, center_idx].cuda()).latent_dist
                    z_2d = posterior.sample()
                    z2d_list.append(z_2d.half() if self.save_precision == 'half' else z_2d.float())
                if self.enable_openclip:
                    _, z_clip = self.openclip_model.visual(self.preprocess_openclip(x_batch[:, :, center_idx].cuda()))
                    z_clip_list.append(z_clip.half() if self.save_precision == 'half' else z_clip.float())
                if return_cframe:
                    cframe_list.append(x_batch[:, :, center_idx])
                 
            z = torch.cat(z_list, dim=0).contiguous().cpu() # n, D, T , H, W
            z_2d = torch.cat(z2d_list, dim=0).contiguous().cpu() if self.enable_2d else None#n, D, H', W'
            z_clip = torch.cat(z_clip_list, dim=0).contiguous().cpu()  if self.enable_openclip else None
            x_cframe = torch.cat(cframe_list, dim=0).contiguous().cpu() if return_cframe else None
        return z, z_clip, z_2d, x_cframe

    def decode_video(self, x, x_2d):
        x = x.unbind(0)
        x_2d = x_2d.unbind(0)
        xrec = []
        for idx in range(len(x)):
          with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
              x0_feats = self.vae3d.encode_2d(x_2d[idx].unsqueeze(0).cuda(), use_tiling=True)
              xrec.append(self.vae3d.decode(x[idx].unsqueeze(0).cuda(), x0_feats, use_tiling=True))
        return xrec
        
    def encode_text(self, caption):
        assert self.enable_t5
        with torch.no_grad():
            caption_embs, emb_masks = self.text_embedder.get_text_embeddings([caption])
        return caption_embs, emb_masks
            
def prepare_tfsm(h, w, disable_resize):
    if not disable_resize:
        tsfm_list = [transforms.Resize(h, antialias=True), transforms.CenterCrop((h, w)),]
    else:
        tsfm_list  = []
    tsfm_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    tsfm = transforms.Compose(tsfm_list)
    return tsfm

def get_best_aspect_ratio(aspect_ratio, n):
    """
    return the index of best available aspect ratio for videos
    """
    if (n > aspect_ratio[-1]):
        return len(aspect_ratio) - 1
    elif n < aspect_ratio[0]:
        return 0
    else:
        pos = bisect_left(aspect_ratio, n)
        if aspect_ratio[pos] == n:
            return pos
        elif n > 1.0:
            return pos - 1
        else:
            return pos

# the format of saved safetensors
def save_latent(z, z_clip, z_2d, p_wopostfix, ar, save_type='torch', fps=None, load_nframes=None, meta_info=None):
    if save_type == 'safetensors':
        p_save = p_wopostfix + '.safetensors'
        tensors = {'z': z}
        if z_2d is not None:
            tensors['z_2d'] = z_2d
        if z_clip is not None:
            tensors['z_clip'] = z_clip
        tensors['ar'] = torch.tensor(float(ar))
        if fps is not None:
            tensors['fps'] = torch.tensor(fps)
        if load_nframes is not None:
            tensors['load_nframes'] = torch.tensor(load_nframes)
        if meta_info is not None:
            for key, value in meta_info.items():
                tensors['MetaInfo_'+key] = value
        safetensors.torch.save_file(tensors, p_save)
    else:
        raise NotImplementedError

def check_exist(p_wopostfix, savetype='torch'):
    if savetype == 'safetensors':
        p_save = p_wopostfix + '.safetensors'
    else:
        raise ValueError('Unknown save type: {}'.format(savetype))
    return os.path.exists(p_save)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_aspect_ratio(ASPECT_RATIO_TYPE):
    if ASPECT_RATIO_TYPE == '1024':
        return ASPECT_RATIO_1024
    elif ASPECT_RATIO_TYPE == '512':
        return ASPECT_RATIO_512
    elif ASPECT_RATIO_TYPE == '256':
        return ASPECT_RATIO_256
    else:
        raise ValueError('Aspect ratio type not supported: {}'.format(autocast_dtype))
        
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--vae2d_type', type=str, default='stabilityai/sd-vae-ft-ema')
  parser.add_argument('--autocast_dtype', type=str, default='fp32', choices=['bf16', 'fp16', 'fp32'])
  parser.add_argument('--vae3d_config', type=str, default=None)
  parser.add_argument('--vae3d_p_ckpt', type=str, default=None)
  parser.add_argument('--vae2d_p_ckpt', type=str, default=None)
  parser.add_argument('--vae3d_nframes', type=int, default=16)
  parser.add_argument('--vae3d_del_decoder', type=str2bool, default=False, help='not loading decoder of Reducio-VAE during extracting latents')
  parser.add_argument('--load_nframes', type=int, default=16, help='the total number of frames loaded for each video')
  parser.add_argument('--p_meta', type=str, help='path to metadata of video dataset')
  parser.add_argument('--dataroot', type=str, help='path to video dataset')
  parser.add_argument('--datasave', type=str, default='output/', help='path to saved video latents')
  parser.add_argument('--text_datasave', type=str, default='output/',  help='path to saved text latents')
  parser.add_argument('--video_datasave', type=str, default='output/', help='path to saved reconstructed video')
  parser.add_argument('--width', type=int, default=512, help='width of input videos into VAE, invalid when aspect_ratio_type is set')
  parser.add_argument('--height', type=int, default=512, help='height of input videos into VAE, invalid when aspect_ratio_type is set')
  parser.add_argument('--aspect_ratio_type', default=None, type=str, choices=['1024', '512', '256'])
  parser.add_argument('--ar', default=None, type=str, help='specify a certain aspect ratio to use')
  parser.add_argument('--fps', type=int, default=16)
  parser.add_argument('--save_type', type=str,default='safetensors', choices=['safetensors'])
  parser.add_argument('--save_precision', type=str, default='half', choices=['float', 'half'])
  parser.add_argument('--n_chunks', type=int, default=1, help='dividing current video dataset into n_chunks')
  parser.add_argument('--chunk_idx', type=int, default=0, help='index of chunks processed in the current run')
  parser.add_argument('--enable_2d', action='store_true', default=False, help='extract the 2d vae fea for Reducio-DiT training')
  parser.add_argument('--enable_openclip', action='store_true', default=False, help='extract the openclip fea for Reducio-DiT training')
  parser.add_argument('--enable_t5', action='store_true', default=False, help='extract the t5 text fea for Reducio-DiT training')
  parser.add_argument('--save_latent', type=str2bool, default=True)
  parser.add_argument('--visualize', type=str2bool, default=True)
  parser.add_argument('--save_vid_type', type=str,default='gif', choices=['gif', 'mp4', 'png'])
  parser.add_argument('--ignore_existed', type=str2bool, default=True)
  parser.add_argument('--max_longedge_of_load', type=int, default=2048)
  args = parser.parse_args()

  extractor = LatentExtractor(
      enable_2d=args.enable_2d,
      enable_openclip=args.enable_openclip,
      enable_t5=args.enable_t5,
      size = (args.height, args.width),
      save_precision=args.save_precision,
      autocast_dtype=args.autocast_dtype,
      aspect_ratio_key=get_aspect_ratio(args.aspect_ratio_type),
      ar=args.ar, 
      vae3d_config=args.vae3d_config,
      vae3d_p_ckpt=args.vae3d_p_ckpt, 
      vae3d_nframes=args.vae3d_nframes,
      vae3d_del_decoder=args.vae3d_del_decoder,
      vae2d_p_ckpt=args.vae2d_p_ckpt,
      load_nframes = args.load_nframes,
      vae2d_type=args.vae2d_type,
  )
  p_meta = args.p_meta
  dataroot = args.dataroot
  load_nframes = args.load_nframes
  os.makedirs(args.datasave, exist_ok=True)
  meta = pd.read_csv(p_meta,on_bad_lines="skip",encoding="ISO-8859-1", engine="python", sep=",")
  print('Total number of videos:', len(meta))
  assert args.n_chunks > 0, 'n_chunks must be greater than 0'
  if args.n_chunks > 1:
    n_chunks, chunk_idx = args.n_chunks, args.chunk_idx
    chunk_intervals = np.linspace(0, len(meta), n_chunks+1, dtype=int)
    chunk_metas = [meta.iloc[chunk_intervals[i]:chunk_intervals[i+1]] for i in range(n_chunks)]
    meta = chunk_metas[chunk_idx]
    print('Processing chunk [{}/{}]'.format(chunk_idx, n_chunks))
    print('Number of videos in chunk:', len(meta))

  idx_count = 0
  ar_list =[]
  meta.loc[:, 'ar'] = '0.0'
  start_idx = 0
  with torch.no_grad():
    for idx, row in tqdm.tqdm(meta.iterrows(), total=len(meta)):
    #   try:
        videoid = str(row['videoid'])
        caption = row['name']
        if videoid.lower().endswith('.mov') or videoid.lower().endswith('.mp4'):
          p_video = os.path.join(dataroot, str(videoid))
        else:
          p_video = os.path.join(dataroot, str(videoid) + '.mp4')
        if not os.path.exists(p_video):
          print('File not found, skipped. Path: {}'.format(p_video))
          continue
        filename = '_'.join(videoid.split('/'))
        if args.ignore_existed and check_exist(os.path.join(args.datasave, filename), args.save_type):
          print('latents exist found, skipped. Path: {}'.format(os.path.join(args.datasave, filename)))
          start_idx = idx + 1
          continue
        # try:
        vr = decord.VideoReader(p_video, ctx=decord.cpu(0))
        height, width, _ = vr[0].shape
        height_ori, width_ori = height, width
        if args.max_longedge_of_load > 0 and max(height, width) > args.max_longedge_of_load:
            shrink_ratio = args.max_longedge_of_load / max(height, width)
            height, width = int(height * shrink_ratio), int(width * shrink_ratio)
        else:
            height, width = -1, -1
            vr = decord.VideoReader(p_video, ctx=decord.cpu(0), width=width, height=height)
            load_nframes_orivideo = len(vr)
            fix_start = max(0, load_nframes_orivideo // 2 - load_nframes // 2)
            raw_fps = vr.get_avg_fps()
            acc_samples = min(load_nframes, load_nframes_orivideo)
            interval = round(raw_fps / args.fps)
            needed_frames = (acc_samples - 1) * interval
            start = 0 if load_nframes_orivideo - needed_frames - 1 < 0 else random.randint(0, load_nframes_orivideo - needed_frames - 1)
            frame_idxs = np.linspace(start=start, stop=min(load_nframes_orivideo - 1, start + needed_frames), num=acc_samples, dtype=int)
            frames = vr.get_batch(frame_idxs)
            frames = frames.float() / 255
            frames = frames.permute(0, 3, 1, 2)
            n_videoframes = frames.shape[0]
        if height == -1:
            height, width = height_ori, width_ori
            meta_info = {'load_nframes_orivideo': load_nframes_orivideo, 'n_videoframes': n_videoframes,  'fps': args.fps, 'height_original': height_ori, 'width_original': width_ori, 'frame_idxs': frame_idxs,}
            meta_info = {k: torch.tensor(v) for k, v in meta_info.items()}
        # except:
        #   print('Error in loading video, skipped. Path: {}'.format(p_video))
        #   continue
        meta_info['height_raw'] = torch.tensor(frames.shape[2])
        meta_info['width_raw'] = torch.tensor(frames.shape[3])
        if args.enable_t5:
          text_save_path = os.path.join(args.text_datasave, filename) + '.safetensors'
          if not args.ignore_existed or not os.path.exists(text_save_path):
            caption_embs, emb_masks = extractor.encode_text(caption)
            text_tensors = {'caption_embs': caption_embs, 'emb_masks': emb_masks}
            safetensors.torch.save_file(text_tensors, text_save_path)
          else:
            print('text latents exist found, skipped. Path: {}'.format(text_save_path))
        x, ar = extractor.preprocess_vae(frames)    
        x = x.permute(1, 0, 2, 3).unsqueeze(0)  # B,C,T,H,W
        visualize_video = not args.vae3d_del_decoder and args.visualize
        z, z_clip, z_2d, x_cframe = extractor.encode_video(x, return_cframe=visualize_video)
        if args.save_latent:
            save_latent(z, z_clip, z_2d, os.path.join(args.datasave, filename), ar, save_type=args.save_type, fps=args.fps, load_nframes=n_videoframes, meta_info=meta_info)
        if visualize_video:
          xrec = extractor.decode_video(z, x_cframe)
          save_tensor_to_video(x, os.path.join(args.video_datasave, f'vid_{filename}_ori'), save_type=args.save_vid_type, duration=1 / args.fps)
          save_tensor_to_video(xrec, os.path.join(args.video_datasave, f'vid_{filename}_rec'), save_type=args.save_vid_type, duration=1 / args.fps)
    #   except:
    #     print('Unknown error, skipped. Path: {}'.format(p_video))
    #     continue


