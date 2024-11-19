import os
import torch
import pandas as pd
import numpy as np
from .base_video_dataset import TextVideoDataset
from torch.utils.data import IterableDataset
from torchvision.transforms import InterpolationMode
from torchvision import transforms

class WebVid(TextVideoDataset):
    """
    WebVid Dataset.
    Assumes webvid data is structured as follows.
    Webvid/
        videos/
            000001_000050/      ($page_dir)
                1.mp4           (videoid.mp4)
                ...
                5000.mp4
            ...
    """

    def _load_metadata(self):
        assert self.metadata_folder_name is not None
        assert self.cut is not None
        metadata_dir = os.path.join(
            self.metadata_dir, self.metadata_folder_name
        )  #'metadata'
        if self.key is None:  # add key to control file we use
            metadata_fp = os.path.join(
                metadata_dir, f"results_{self.cut}_{self.split}.csv"
            )
        else:
            metadata_fp = os.path.join(
                metadata_dir, f"results_{self.cut}_{self.split}_{self.key}.csv"
            )
        print(metadata_fp)
        metadata = pd.read_csv(
            metadata_fp, on_bad_lines="skip",encoding="ISO-8859-1",engine="python",sep=','
        )  # fix csv file error

        if self.subsample < 1:
            metadata = metadata.sample(frac=self.subsample)
        elif self.split == "val":
            try:
                metadata = metadata.sample(
                    1000, random_state=0
                )  # 15k val is unnecessarily large, downsample.
            except:
                print(
                    "there are less than 1000 samples in the val set, thus no downsampling is done"
                )
                pass

        metadata["caption"] = metadata["name"]
        del metadata["name"]
        self.metadata = metadata
        # TODO: clean final csv so this isn't necessary
        self.metadata.dropna(inplace=True)
        # self.metadata['caption'] = self.metadata['caption'].str[:350]

    def _get_video_path(self, sample):
        # rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        videoid =  str(sample["videoid"]) 
        if videoid.lower().endswith('.mov') or videoid.lower().endswith('.mp4'):
            rel_video_fp = videoid
        else:
            rel_video_fp = videoid + ".mp4"
        full_video_fp = os.path.join(self.data_dir, rel_video_fp)
        return full_video_fp, rel_video_fp

    def _get_caption(self, sample):
        return sample["caption"]
    
    
def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR

