# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import sys
import argparse
from collections import defaultdict
import logging
import os
import glob
import pickle
import random

import numpy as np
from pesq import pesq
import speechmetrics

from tqdm import tqdm

import torch
import torchaudio
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F

from speechbrain.utils.metric_stats import ErrorRateStats, EER, minDCF
from speechbrain.pretrained import EncoderDecoderASR, SpeakerRecognition

from vida.dataset import compute_spectrogram, to_tensor,normalize
from vida.predictor import Predictor
from vida.trainer import griffinlim
from vida.utils import overlap_chunk, load_spkcfg, load_noise, parse_csv, compute_scores

EPS = 1e-7

# NOTE: Load the Vida model and ASR models from speechbrain
def load_model(args, checkpoint_path, device):
    if args.est_pred:
        dereverber = Predictor(input_channel=args.num_channel, use_rgb=args.use_rgb, use_depth=args.use_depth,
                              no_mask=args.no_mask, limited_fov=args.limited_fov,
                              mean_pool_visual=args.mean_pool_visual).to(device=device)
        dereverber.load_state_dict(torch.load(checkpoint_path, map_location=device)['predictor'])
        dereverber.eval()
    else:
        dereverber = None

    if args.eval_asr:
        asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech",
                                                savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
                                                run_opts={"device":"cuda:0"})
    else:
        asr_model = None

    if args.eval_spkrec:
        verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                                       savedir="pretrained_models/spkrec-ecapa-voxceleb",
                                                       run_opts={"device":"cuda:0"})
    else:
        verification = None

    return dereverber, asr_model, verification

def dereverberate(dereverber, input_data, device, args):
    """
        dereverb the data, return the waveform or spectrogram
    """
    # Prepare the input
    tgt_shape = (256, 256)
    recv_audio = input_data['receiver_audio']
    recv_audio = to_tensor(recv_audio).to(device, dtype=torch.float)
    recv_audio = normalize(recv_audio, norm='peak')

    visual_sensors = []
    inputs = dict()
    if args.use_rgb:
        inputs['rgb'] = to_tensor(np.concatenate([x / 255.0 for x in input_data['rgb']], axis=1)).permute(2, 0, 1)
        visual_sensors.append('rgb')
    if args.use_depth:
        inputs['depth'] = to_tensor(np.concatenate(input_data['depth'], axis=1)).unsqueeze(0)
        visual_sensors.append('depth')
    for visual_sensor in visual_sensors:
        inputs[visual_sensor] = inputs[visual_sensor].to(device=device, dtype=torch.float)

    if args.limited_fov:
        for visual_sensor in visual_sensors:
            inputs[visual_sensor] = inputs[visual_sensor][:, :, :256]
            if args.crop:
                inputs[visual_sensor] = torchvision.transforms.CenterCrop((336, 224))(inputs[visual_sensor])
    else:
        for visual_sensor in visual_sensors:
            inputs[visual_sensor] = torchvision.transforms.Resize((192, 576))(inputs[visual_sensor])
            if args.crop:
                inputs[visual_sensor] = torchvision.transforms.CenterCrop((168, 504))(inputs[visual_sensor])

    win = torch.hamming_window(400).to(device, dtype=torch.float)
    if args.use_real_imag:
        recv_spec = compute_spectrogram(recv_audio, log=False, use_complex=True)
    else:
        recv_spec = torch.cat([compute_spectrogram(recv_audio, log=False, use_mag=True),
                               compute_spectrogram(recv_audio, log=False, use_phase=True)], dim=-1)

    T = recv_spec.size(1)
    size, step, left_padding = tgt_shape[1], tgt_shape[1] // 2, tgt_shape[1] // 4
    overlapped_spec = overlap_chunk(recv_spec[1: tgt_shape[0] + 1, :, :].permute(2, 0, 1), 2, size, step,
                                    left_padding)

    overlapped_spec = overlapped_spec.permute(2, 0, 1, 3)
    B = overlapped_spec.size(0)
    for visual_sensor in visual_sensors:
        inputs[visual_sensor] = inputs[visual_sensor].unsqueeze(0).repeat(B, 1, 1, 1)

    input_spec = overlapped_spec[:, :args.num_channel, ...]
    if args.log_mag:
        assert not args.use_real_imag
        if args.log1p:
            input_spec[:, 0, ...] = torch.log1p(input_spec[:, 0, ...])
        else:
            input_spec[:, 0, ...] = torch.log10(input_spec[:, 0, ...] + EPS)

    # perform dereverb
    with torch.no_grad():
        inputs.update({'spectrograms': input_spec, 'distance': None})
        output = dereverber(inputs)

    # post process output 
    pred_mask = output['pred_mask']
    if args.no_mask:
        pred_spec = pred_mask
    elif args.num_channel == 1:
        pred_spec = pred_mask * input_spec
    else:
        if args.use_real_imag:
            pred_spec_real = input_spec[:, 0, :, :] * pred_mask[:, 0, :, :] \
                             - input_spec[:, 1, :, :] * pred_mask[:, 1, :, :]
            pred_spec_img = input_spec[:, 0, :, :] * pred_mask[:, 1, :, :] \
                            + input_spec[:, 1, :, :] * pred_mask[:, 0, :, :]
            pred_spec = torch.cat((pred_spec_real.unsqueeze(1), pred_spec_img.unsqueeze(1)), 1)
        else:
            pred_spec_mag = input_spec[:, 0, :, :] * pred_mask[:, 0, :, :]
            pred_spec_phase = input_spec[:, 1, :, :] * pred_mask[:, 1, :, :]
            pred_spec = torch.cat((pred_spec_mag.unsqueeze(1), pred_spec_phase.unsqueeze(1)), 1)

    if args.log_mag:
        if args.log1p:
            pred_spec[:, 0, ...] = torch.pow(np.e, pred_spec[:, 0, ...]) - 1
        else:
            pred_spec[:, 0, ...] = torch.pow(10, pred_spec[:, 0, ...])

    pred_spec = pred_spec[:, :, :, left_padding:left_padding + step].permute(2, 0, 3, 1).reshape(
        tgt_shape[0], -1, args.num_channel)
    full_pred_spec = recv_spec.clone()
    full_pred_spec[1: tgt_shape[0] + 1, :, :args.num_channel] = pred_spec[:, :T, :]

    if args.use_real_imag:
        power_spec = full_pred_spec.pow(2).sum(-1).unsqueeze(0)
        pred_audio = torch.istft(full_pred_spec, n_fft=512, hop_length=160, win_length=400, window=win).unsqueeze(0)
    else:
        power_spec = full_pred_spec[..., 0].pow(2).unsqueeze(0)
        pred_audio = griffinlim(full_pred_spec[..., 0].unsqueeze(0), None, #full_pred_spec[..., 1].unsqueeze(0),
                                torch.hamming_window(400, device=device), n_fft=512, hop_length=160, win_length=400,
                                power=1, n_iter=30, rand_init=True)

    return power_spec, pred_audio

def eval_batch(args, files, checkpoint_path, device, data_list, sr=16000):
    """
    read the PKL file, perform dereveber and ASR
    """
    dereverber, asr_model, verification = load_model(args, checkpoint_path, device)

    running_metrics = defaultdict(list)
    count = 0
    if args.eval_asr:
        cer_stats = ErrorRateStats()
        test_id_list = parse_csv(data_list)
    if args.eval_spkrec:
        enhance_cache = {}
    if args.eval_dereverb:
        metrics = speechmetrics.load(['stoi'])

    if args.use_noise:
        noise_pool = load_noise(args.split)

    for file in tqdm(files):
        speech_id = os.path.basename(file)
        speech_id = speech_id.replace('.pkl', '')
        scene_id = file.split('/')[-2]
        non_duplicate_speech_id = f'{speech_id}_{scene_id}'

        count += 1
        with open(file, 'rb') as f:
            input_data = pickle.load(f)     
        source_audio = to_tensor(input_data['source_audio'])
        receiver_audio = to_tensor(input_data['receiver_audio'])

        if args.use_noise:
            waveform_length = receiver_audio.shape[0]
            noise_start_index = np.random.randint(0, noise_pool.shape[0] - waveform_length)
            noise = noise_pool[noise_start_index: noise_start_index + waveform_length]
            noise_energy = 10 * torch.log10(torch.sum(noise ** 2))
            signal_energy = 10 * torch.log10(torch.sum(receiver_audio ** 2))
            weight = torch.pow(10, ((signal_energy - args.snr) - noise_energy) / 20)
            receiver_audio += noise * weight

        if args.est_pred:
            pred_spec, enhanced_audio = dereverberate(dereverber, input_data, device, args)
        else:
            if args.use_clean:
                enhanced_audio = source_audio.unsqueeze(0)
            else:
                enhanced_audio = receiver_audio.unsqueeze(0)
            pred_spec = torch.stft(enhanced_audio[0], n_fft=512, hop_length=160, win_length=400,
                                   window=torch.hamming_window(400), pad_mode='constant'). \
                pow(2).sum(-1).unsqueeze(0).to(device)

        if args.eval_spkrec:
            enhance_cache[non_duplicate_speech_id] = [source_audio.cuda(), enhanced_audio.cpu()]

        if args.eval_dereverb:
            reference = source_audio.numpy()
            enhanced = enhanced_audio.cpu()[0].numpy()
            stoi_score = metrics(enhanced, reference, rate=16000)['stoi'][0]
            pesq_score = pesq(16000, reference, enhanced, 'wb')
            running_metrics['stoi'].append(stoi_score)
            running_metrics['pesq'].append(pesq_score)

        if args.eval_asr:
            # if args.num_channel == 1:
            #     pred, tokens = asr_model.transcribe_batch_spectrogram(pred_spec, torch.tensor([1.0]))
            # else:
            pred, tokens = asr_model.transcribe_batch(enhanced_audio, torch.tensor([1.0]))
            
            _, target, _ = test_id_list[speech_id]
            cer_stats.append(ids=[speech_id], predict=np.array([pred[0].split(' ')]),
                             target=np.array([target.split(' ')]))
            if count % args.print_frequency == 0:
                print(f"Current WER for first {count} sentences:", cer_stats.summarize()['WER'])

        if args.save_audio:
            audio_dir = os.path.join(args.model_dir, args.split, non_duplicate_speech_id)
            os.makedirs(audio_dir, exist_ok=True)
            pred_file = os.path.join(audio_dir, f'pred.wav')

            torchaudio.save(pred_file, normalize(enhanced_audio.cpu()), sr)

    if args.eval_dereverb:
        for metric, values in running_metrics.items():
            avg_metric_value = np.mean(values)
            print(metric, avg_metric_value)

    if args.eval_asr:
        wer = cer_stats.summarize()['WER']
        print(f"Final WER:", wer)

    if args.eval_spkrec:
        pos_pairs, neg_pairs = load_spkcfg(files, args.split)
        print(f"num pos pairs:{len(pos_pairs)}, num neg pairs:{len(neg_pairs)}")
        negative_scores = compute_scores(neg_pairs, verification, enhance_cache).cpu()
        positive_scores = compute_scores(pos_pairs, verification, enhance_cache).cpu()

        eer, th = EER(positive_scores, negative_scores)
        min_dcf, th = minDCF(positive_scores, negative_scores)
        print(f"EER:{eer}, min_dcf: {min_dcf}")

def main():
    parser = argparse.ArgumentParser("Evaluate WER of dataset")
    parser.add_argument('--print-frequency', type=int, default=10, help='the frequency to print current WER')
    parser.add_argument('--librispeech', type=str, default='data/LibriSpeech/')
    parser.add_argument('--split', type=str, default='test-unseen', choices=['train', 'test-unseen', 'val'])
    parser.add_argument("--use-rgb", default=False, action='store_true')
    parser.add_argument("--use-depth", default=False, action='store_true')
    parser.add_argument("--use-rgbd", default=False, action='store_true')
    parser.add_argument("--num-channel", default=1, type=int)
    parser.add_argument("--use-noise", default=False, action='store_true')
    parser.add_argument("--snr", default=20, type=float)
    parser.add_argument('--model-dir', type=str, default='data/models/vida')    
    parser.add_argument('--pretrained-path', type=str, default='')
    parser.add_argument('--ckpt', type=int, default=-1)
    parser.add_argument('--est-pred', default=False, action='store_true')
    parser.add_argument('--use-clean', default=False, action='store_true')
    parser.add_argument('--eval-dereverb', default=False, action='store_true')
    parser.add_argument('--eval-asr', default=False, action='store_true')
    parser.add_argument('--eval-spkrec', default=False, action='store_true')
    parser.add_argument('--log-mag', default=False, action='store_true')
    parser.add_argument('--log1p', default=False, action='store_true')
    parser.add_argument('--use-real-imag', default=False, action='store_true')
    parser.add_argument('--no-mask', default=False, action='store_true')
    parser.add_argument('--save-audio', default=False, action='store_true')
    parser.add_argument('--limited-fov', default=False, action='store_true')
    parser.add_argument('--crop', default=False, action='store_true')
    parser.add_argument('--normalize', default=False, action='store_true')
    parser.add_argument('--mean-pool-visual', default=False, action='store_true')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")
    
    args = parser.parse_args()
    if args.split == 'train':
        data_list = f'{args.librispeech}/train-clean-360.csv'
    elif args.split.startswith('val'):
        data_list = f'{args.librispeech}/dev-clean.csv'
    elif args.split.startswith('test'):
        data_list = f'{args.librispeech}/test-clean.csv'
    else:
        raise ValueError

    random.seed(0)
    np.random.seed(0)

    device = torch.device("cuda", 0)

    data_dir = os.path.join('data', args.split)
    files = sorted(glob.glob(data_dir + '/**/*.pkl', recursive=True))
    eval_batch(args, files, args.pretrained_path, device, data_list)

if __name__ == '__main__':
    main()