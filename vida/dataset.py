# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import glob
import time
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchaudio
from torch.utils.data import Dataset
from scipy.signal import fftconvolve
from pyroomacoustics.experimental.rt60 import measure_rt60


def compute_spectrogram(audio_data, log=False, use_mag=False, use_phase=False, use_complex=False):
    def safe_log10(x, eps=1e-10):
        result = np.where(x > eps, x, -10)

        np.log10(result, out=result, where=result > 0)
        return result

    audio_data = to_tensor(audio_data)
    stft = torch.stft(audio_data, n_fft=512, hop_length=160, win_length=400,
                      window=torch.hamming_window(400, device=audio_data.device), pad_mode='constant',
                      return_complex=not use_complex)

    if use_mag:
        spectrogram = stft.abs().unsqueeze(-1)
    elif use_phase:
        spectrogram = stft.angle().unsqueeze(-1)
    elif use_complex:
        # one channel for real and one channel for imaginary
        spectrogram = stft
    else:
        raise ValueError

    return 


def to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


class SoundSpacesSpeechDataset(Dataset):
    def __init__(self, split, normalize_whole=True, normalize_segment=False, use_real_imag=False, 
                 use_rgb=False, use_depth=False, limited_fov=False, use_rgbd=False, hop_length=160, 
                 deterministic_eval=False):
        self.split = split
        self.data_dir = os.path.join('data/soundspaces_speech', split)
        self.files = glob.glob(self.data_dir + '/**/*.pkl', recursive=True)
        np.random.shuffle(self.files)
        self.normalize_whole = normalize_whole
        self.normalize_segment = normalize_segment
        self.use_real_imag = use_real_imag
        self.use_rgb = use_rgb or use_rgbd
        self.use_depth = use_depth or use_rgbd
        self.limited_fov = limited_fov
        self.hop_length = hop_length
        self.deterministic_eval = deterministic_eval

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file = self.files[item]
        with open(file, 'rb') as fo:
            data = pickle.load(fo)

        receiver_audio = data['receiver_audio']
        source_audio = data['source_audio']

        src_spec, recv_spec, src_wav, recv_wav = self.process_audio(source_audio, receiver_audio)

        sample = dict()
        # sample['src_wav'] = np.pad(source_audio, (0, 16000 * 30 - source_audio.shape[0]))
        sample['src_wav'] = src_wav
        sample['recv_wav'] = recv_wav
        sample['src_spec'] = src_spec
        sample['recv_spec'] = recv_spec

        # stitch images into panorama
        visual_sensors = []
        if self.use_rgb:
            sample['rgb'] = to_tensor(np.concatenate([x / 255.0 for x in data['rgb']], axis=1)).permute(2, 0, 1)
            visual_sensors.append('rgb')
        if self.use_depth:
            sample['depth'] = to_tensor(np.concatenate(data['depth'], axis=1)).unsqueeze(0)
            visual_sensors.append('depth')

        if len(visual_sensors) > 0:
            if self.split == 'train':
                # data augmentation
                width_shift = None
                for visual_sensor in visual_sensors:
                    if width_shift is None:
                        width_shift = np.random.randint(0, sample[visual_sensor].shape[-1])
                    sample[visual_sensor] = torch.roll(sample[visual_sensor], width_shift, dims=-1)

            if self.limited_fov:
                # crop image to size 384 * 256
                if self.split == 'train':
                    offset = None
                    for visual_sensor in visual_sensors:
                        if offset is None:
                            offset = np.random.randint(0, sample[visual_sensor].shape[-1] - 256)
                        sample[visual_sensor] = sample[visual_sensor][:, :, offset: offset + 256]
                else:
                    for visual_sensor in visual_sensors:
                        sample[visual_sensor] = sample[visual_sensor][:, :, :256]
            else:
                for visual_sensor in visual_sensors:
                    sample[visual_sensor] = torchvision.transforms.Resize((192, 576))(sample[visual_sensor])

        return sample

    def process_audio(self, source_audio, receiver_audio):
        # normalize the intensity before padding
        hop_length = self.hop_length
        if hop_length == 128:
            target_shape = (256, 320)
        else:
            target_shape = (256, 256)
        waveform_length = target_shape[1] * hop_length

        if self.normalize_whole:
            receiver_audio = normalize(receiver_audio, norm='peak')
            source_audio = normalize(source_audio, norm='peak')

        # pad audio
        if target_shape[1] * hop_length > source_audio.shape[0]:
            receiver_audio = np.pad(receiver_audio, (0, max(0, waveform_length - receiver_audio.shape[0])))
            source_audio = np.pad(source_audio, (0, waveform_length - source_audio.shape[0]))

        if self.deterministic_eval and self.split != 'train':
            start_index = 0
        else:
            start_index = np.random.randint(0, source_audio.shape[0] - waveform_length) \
                if source_audio.shape[0] != waveform_length else 0
        source_clip = source_audio[start_index: start_index + waveform_length]
        receiver_clip = receiver_audio[start_index: start_index + waveform_length]

        # normalize the short clip
        if self.normalize_segment:
            source_clip = normalize(source_clip, norm='peak')
            receiver_clip = normalize(receiver_clip, norm='peak')

        if not self.use_real_imag:
            stft = torch.stft(to_tensor(source_clip), n_fft=512, hop_length=hop_length, win_length=400,
                              window=torch.hamming_window(400), pad_mode='constant', return_complex=True)
            src_spec = torch.stack([stft.abs(), stft.angle()], dim=-1)
            stft = torch.stft(to_tensor(receiver_clip), n_fft=512, hop_length=hop_length, win_length=400,
                              window=torch.hamming_window(400), pad_mode='constant', return_complex=True)
            recv_spec = torch.stack([stft.abs(), stft.angle()], dim=-1)
        else:
            src_spec = torch.stft(to_tensor(source_clip), n_fft=512, hop_length=hop_length, win_length=400,
                                  window=torch.hamming_window(400), pad_mode='constant', return_complex=False)
            recv_spec = torch.stft(to_tensor(receiver_clip), n_fft=512, hop_length=hop_length, win_length=400,
                                   window=torch.hamming_window(400), pad_mode='constant', return_complex=False)
        return src_spec, recv_spec, source_clip, receiver_clip


def normalize(audio, norm='peak'):
    if norm == 'peak':
        peak = abs(audio).max()
        if peak != 0:
            return audio / peak
        else:
            return audio
    elif norm == 'rms':
        if torch.is_tensor(audio):
            audio = audio.numpy()
        audio_without_padding = np.trim_zeros(audio, trim='b')
        rms = np.sqrt(np.mean(np.square(audio_without_padding))) * 100
        if rms != 0:
            return audio / rms
        else:
            return audio
    else:
        raise NotImplementedError
