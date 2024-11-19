import os
import random
from abc import abstractmethod

import av
import cv2
import decord
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms


# def init_transform_dict(input_res=224,
def init_transform_dict(
    input_res_h=224,
    input_res_w=224,
    randcrop_scale=(0.5, 1.0),
    color_jitter=(0, 0, 0),
    norm_mean=(0.5, 0.5, 0.5),
    norm_std=(0.5, 0.5, 0.5),
):
    # todo: This part need to be discussed and designed carefully.
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (input_res_h, input_res_w), scale=randcrop_scale,antialias=True
                ),
                normalize,
            ]
        ),
        "val": transforms.Compose(
            [
                # todo: should we use crop for validation and test?
                transforms.Resize((input_res_h, input_res_w),antialias=True),
                normalize,
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((input_res_h, input_res_w),antialias=True),
                normalize,
            ]
        ),
    }
    return tsfm_dict


class TextVideoDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        text_params,
        video_params,
        data_dir,
        metadata_dir=None,
        metadata_folder_name=None,  # "webvid10m_meta",
        split="train",
        tsfms=None,
        cut=None,
        key=None,
        subsample=1,
        sliding_window_stride=-1,
        reader="decord",
        first_stage_key="video",
        cond_stage_key="txt",
        skip_missing_files=True,
    ):
        # print(dataset_name, text_params, video_params)
        # WebVid {'input': 'text'} {'input_res': 224, 'num_frames': 1, 'loading': 'lax'}
        self.dataset_name = dataset_name
        self.text_params = text_params
        self.video_params = video_params
        # check for environment variables
        self.data_dir = os.path.expandvars(data_dir)
        if metadata_dir is not None:
            self.metadata_dir = os.path.expandvars(metadata_dir)
        else:
            self.metadata_dir = self.data_dir
        # added parameters
        self.metadata_folder_name = metadata_folder_name
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.skip = skip_missing_files
        self.lack_files = []
        self.split = split
        self.key = key
        tsfm_params = (
            {}
            if "tsfm_params" not in video_params.keys()
            else video_params["tsfm_params"]
        )
        # tsfm_params['input_res'] = video_params['input_res']
        if "input_res_h" in video_params and 'input_res_w' in video_params:
            tsfm_params["input_res_h"] = video_params["input_res_h"]
            tsfm_params["input_res_w"] = video_params["input_res_w"]
        elif 'input_res' in video_params:
            self.video_params["input_res_h"] = video_params["input_res"]
            self.video_params["input_res_w"] = video_params["input_res"]
            tsfm_params["input_res_h"] = video_params["input_res"]
            tsfm_params["input_res_w"] = video_params["input_res"]
            
        tsfm_dict = init_transform_dict(**tsfm_params)

        split = "train" if "train" in split else split
        tsfms = tsfm_dict[split]

        self.transforms = tsfms
        self.cut = cut
        self.subsample = subsample
        self.sliding_window_stride = sliding_window_stride
        self.video_reader = video_reader[reader]
        self.label_type = "caption"
        self.frame_sample = video_params.get("frame_sample", "proportional")
        self._load_metadata()
        if self.sliding_window_stride != -1:
            if self.split != "test":
                raise ValueError(
                    "Fixing frame sampling is for test time only. can remove but..."
                )
            self._fix_temporal_samples()

    @abstractmethod
    def _load_metadata(self):
        raise NotImplementedError("Metadata loading must be implemented by subclass")

    @abstractmethod
    def _get_video_path(self, sample):
        raise NotImplementedError(
            "Get video path function must be implemented by subclass"
        )

    def _get_caption(self, sample):
        raise NotImplementedError(
            "Get caption function must be implemented by subclass"
        )

    def _get_video_lens(self):
        vlen_li = []
        for idx, row in self.metadata.iterrows():
            video_path = self._get_video_path(row)[0]
            vlen_li.append(get_video_len(video_path))

        return vlen_li

    def _fix_temporal_samples(self):
        self.metadata["vlen"] = self._get_video_lens()
        self.metadata["frame_intervals"] = self.metadata["vlen"].apply(
            lambda x: np.linspace(
                start=0, stop=x, num=min(x, self.video_params["num_frames"]) + 1
            ).astype(int)
        )
        self.metadata["fix_start"] = self.metadata["frame_intervals"].apply(
            lambda x: np.arange(0, int(x[-1] / len(x - 1)), self.sliding_window_stride)
        )
        self.metadata = self.metadata.explode("fix_start")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        # if not os.path.exists(video_fp):
        #     return self.__getitem__(np.random.choice(self.__len__()))
        caption = self._get_caption(sample)

        video_loading = self.video_params.get("loading", "strict")  #
        # frame_sample = 'rand'
        fix_start = None
        # if self.split == 'test':
        #     frame_sample = 'uniform'
        if self.sliding_window_stride != -1:
            fix_start = sample["fix_start"]

        # try:
        if os.path.isfile(video_fp):
            # imgs, idxs = self.video_reader(video_fp, self.video_params['num_frames'], frame_sample,
            #                                fix_start=fix_start)
            if self.frame_sample == "equally spaced":
                sample_factor = self.video_params.get("es_interval", 10)
            elif self.frame_sample == "proportional":
                sample_factor = self.video_params.get("prop_factor", 3)
            imgs, idxs = self.video_reader(
                video_fp,
                self.video_params["num_frames"],
                self.frame_sample,
                fix_start=fix_start,
                sample_factor=sample_factor,
            )
        else:
            print_str = f"Warning: missing video file {video_fp}."
            if video_fp not in self.lack_files:
                self.lack_files.append(video_fp)
            if self.skip:
                print_str += " Resampling another video."
                print(print_str)
                return self.__getitem__(np.random.choice(self.__len__()))
            else:
                print(print_str)
                assert False

        # except Exception as e:
        #     if video_loading == "strict":
        #         raise ValueError(
        #             f"Video loading failed for {video_fp}, video loading for this dataset is strict."
        #         ) from e
        #     else:
        #         print("Warning: using the pure black image as the frame sample")
        #         # imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
        #         imgs = Image.new(
        #             "RGB",
        #             (
        #                 self.video_params["input_res_w"],
        #                 self.video_params["input_res_h"],
        #             ),
        #             (0, 0, 0),
        #         )
        #         imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            imgs = self.transforms(imgs)  # normalize or 2 * x - 1 ?

        # final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
        #                      self.video_params['input_res']])
        final = torch.zeros(
            [
                self.video_params["num_frames"],
                3,
                self.video_params["input_res_h"],
                self.video_params["input_res_w"],
            ]
        )

        final[: imgs.shape[0]] = imgs

        meta_arr = {
            "raw_captions": caption,
            "paths": rel_fp,
            "dataset": self.dataset_name,
        }
        data = {
            self.first_stage_key: final,
            self.cond_stage_key: caption,
            "meta": meta_arr,
        }
        return data


class TextImageDataset(TextVideoDataset):
    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        video_loading = self.video_params.get("loading", "strict")

        try:
            img = Image.open(video_fp).convert("RGB")
        except:
            if video_loading == "strict":
                raise ValueError(
                    f"Image loading failed for {video_fp}, image loading for this dataset is strict."
                )
            else:
                # img = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                img = Image.new(
                    "RGB",
                    (
                        self.video_params["input_res_w"],
                        self.video_params["input_res_h"],
                    ),
                    (0, 0, 0),
                )

        # convert to tensor because video transforms don't, expand such that its a 1-frame video.
        img = transforms.ToTensor()(img).unsqueeze(0)
        if self.transforms is not None:
            img = self.transforms(img)
        meta_arr = {
            "raw_captions": caption,
            "paths": rel_fp,
            "dataset": self.dataset_name,
        }
        data = {"video": img, "text": caption, "meta": meta_arr}
        return data


def sample_frames(
    num_frames, vlen, sample="rand", fix_start=None, **kwargs
):  # TBD, what do you need
    """
    num_frames: The number of frames to sample.
    vlen: The length of the video.
    sample: The sampling method.
        choices of frame_sample:
        - 'equally spaced': sample frames equally spaced
            e.g.,1s video has 30 frames, when 'es_interval'=8, we sample frames with spacing of 8
        - 'proportional': sample frames proportional to the length of the frames in one second
            e.g., 1s video has 30 frames, when 'prop_factor'=3, we sample frames with spacing of 30/3=10
        - 'random': sample frames randomly (not recommended)
        - 'uniform': sample frames uniformly (not recommended)
    fix_start: The starting frame index. If it is not None, then it will be used as the starting frame index.
    """
    acc_samples = min(num_frames, vlen)
    if sample in ["rand", "uniform"]:
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
        elif fix_start is not None:
            frame_idxs = [x[0] + fix_start for x in ranges]
        elif sample == "uniform":
            frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    elif sample in ["equally spaced", "proportional"]:
        if sample == "equally spaced":
            raise NotImplementedError  # need to pass in the corresponding parameters
        else:
            interval = round(kwargs["fps"] / kwargs["sample_factor"])
            needed_frames = (acc_samples - 1) * interval

            if fix_start is not None:
                start = fix_start
            else:
                if vlen - needed_frames - 1 < 0:
                    start = 0
                else:
                    start = random.randint(0, vlen - needed_frames - 1)
            frame_idxs = np.linspace(
                start=start, stop=min(vlen - 1, start + needed_frames), num=acc_samples
            ).astype(int)
    else:
        raise NotImplementedError

    return frame_idxs


def read_frames_cv2(video_path, num_frames, sample="rand", fix_start=None, **kwargs):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    # get indexes of sampled frames
    fps = cap.get(cv2.CAP_PROP_FPS)  # not verified yet, might cause bug.
    frame_idxs = sample_frames(
        num_frames,
        vlen,
        sample=sample,
        fix_start=fix_start,
        fps=fps,
        sample_factor=kwargs["sample_factor"],
    )
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')

    frames = torch.stack(frames).float() / 255
    cap.release()
    return frames, success_idxs


def read_frames_av(video_path, num_frames, sample="rand", fix_start=None, **kwargs):
    reader = av.open(video_path)
    try:
        frames = []
        frames = [
            torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)
        ]
    except (RuntimeError, ZeroDivisionError) as exception:
        print(
            "{}: WEBM reader cannot open {}. Empty "
            "list returned.".format(type(exception).__name__, video_path)
        )
    vlen = len(frames)
    # frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    fps = reader.streams.video[0].average_rate  # not verified yet, might cause bug.
    frame_idxs = sample_frames(
        num_frames,
        vlen,
        sample=sample,
        fix_start=fix_start,
        fps=fps,
        sample_factor=kwargs["sample_factor"],
    )
    frames = torch.stack([frames[idx] for idx in frame_idxs]).float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


decord.bridge.set_bridge("torch")


def read_frames_decord(video_path, num_frames, sample="rand", fix_start=None, **kwargs):
    video_reader = decord.VideoReader(video_path, num_threads=0)
    vlen = len(video_reader)
    # frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    fps = video_reader.get_avg_fps()  # note that the fps here is float.
    frame_idxs = sample_frames(
        num_frames,
        vlen,
        sample=sample,
        fix_start=fix_start,
        fps=fps,
        sample_factor=kwargs["sample_factor"],
    )
    frames = video_reader.get_batch(frame_idxs)
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs


def get_video_len(video_path):
    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen


video_reader = {
    "av": read_frames_av,
    "cv2": read_frames_cv2,
    "decord": read_frames_decord,
}
