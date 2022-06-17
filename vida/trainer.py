# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import logging
import copy
import shutil
import random
from collections import defaultdict
from typing import Optional
import math
import glob

import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import speechmetrics
from pesq import pesq
from torch import Tensor
import librosa
import torchvision

from vida.predictor import Predictor
from vida.dataset import SoundSpacesSpeechDataset

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
EPS = 1e-7


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_worker(worker_id):
    random.seed(worker_id)
    np.random.seed(worker_id)


def complex_norm(
        complex_tensor: Tensor,
        power: float = 1.0
) -> Tensor:
    r"""Compute the norm of complex tensor input.

    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
        power (float): Power of the norm. (Default: `1.0`).

    Returns:
        Tensor: Power of the normed input tensor. Shape of `(..., )`
    """

    # Replace by torch.norm once issue is fixed
    # https://github.com/pytorch/pytorch/issues/34279
    return complex_tensor.pow(2.).sum(-1).pow(0.5 * power)


def griffinlim(
        specgram: Tensor,
        phase,
        window: Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        power: float,
        normalized: bool = False,
        n_iter: int = 32,
        momentum: float = 0.99,
        length: Optional[int] = None,
        rand_init: bool = None
) -> Tensor:
    r"""Compute waveform from a linear scale magnitude spectrogram using the Griffin-Lim transformation.
        Implementation ported from `librosa`.

    Args:
        specgram (Tensor): A magnitude-only STFT spectrogram of dimension (..., freq, frames)
            where freq is ``n_fft // 2 + 1``.
        window (Tensor): Window tensor that is applied/multiplied to each frame/window
        n_fft (int): Size of FFT, creates ``n_fft // 2 + 1`` bins
        hop_length (int): Length of hop between STFT windows. (
            Default: ``win_length // 2``)
        win_length (int): Window size. (Default: ``n_fft``)
        power (float): Exponent for the magnitude spectrogram,
            (must be > 0) e.g., 1 for energy, 2 for power, etc.
        normalized (bool): Whether to normalize by magnitude after stft.
        n_iter (int): Number of iteration for phase recovery process.
        momentum (float): The momentum parameter for fast Griffin-Lim.
            Setting this to 0 recovers the original Griffin-Lim method.
            Values near 1 can lead to faster convergence, but above 1 may not converge.
        length (int or None): Array length of the expected output.
        rand_init (bool): Initializes phase randomly if True, to zero otherwise.

    Returns:
        torch.Tensor: waveform of (..., time), where time equals the ``length`` parameter if given.
    """
    assert momentum < 1, 'momentum={} > 1 can be unstable'.format(momentum)
    assert momentum >= 0, 'momentum={} < 0'.format(momentum)

    if normalized:
        warnings.warn(
            "The argument normalized is not used in Griffin-Lim, "
            "and will be removed in v0.9.0 release. To suppress this warning, "
            "please use `normalized=False`.")

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape([-1] + list(shape[-2:]))

    specgram = specgram.pow(1 / power)

    # randomly initialize the phase
    batch, freq, frames = specgram.size()
    if phase is None:
        if rand_init:
            angles = 2 * math.pi * torch.rand(batch, freq, frames)
        else:
            angles = torch.zeros(batch, freq, frames)
        angles = torch.stack([angles.cos(), angles.sin()], dim=-1) \
            .to(dtype=specgram.dtype, device=specgram.device)
    else:
        # use input phase instead of random phase
        angles = torch.stack([phase.cos(), phase.sin()], dim=-1)
    specgram = specgram.unsqueeze(-1).expand_as(angles)

    # And initialize the previous iterate to 0
    rebuilt = torch.tensor(0.)

    for _ in range(n_iter):
        # Store the previous iterate
        tprev = rebuilt

        # Invert with our current estimate of the phases
        inverse = torch.istft(specgram * angles,
                              n_fft=n_fft,
                              hop_length=hop_length,
                              win_length=win_length,
                              window=window,
                              length=length).float()

        # Rebuild the spectrogram
        rebuilt = torch.view_as_real(
            torch.stft(
                input=inverse,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                pad_mode='reflect',
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        )

        # Update our phase estimates
        angles = rebuilt
        if momentum:
            angles = angles - tprev.mul_(momentum / (1 + momentum))
        angles = angles.div(complex_norm(angles).add(1e-16).unsqueeze(-1).expand_as(angles))

    # Return the final phase estimates
    waveform = torch.istft(specgram * angles,
                           n_fft=n_fft,
                           hop_length=hop_length,
                           win_length=win_length,
                           window=window,
                           length=length)

    # unpack batch
    waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])

    return waveform


def search_dict(ckpt_dict, encoder_name):
    encoder_dict = {}
    for key, value in ckpt_dict.items():
        if encoder_name in key:
            encoder_dict['.'.join(key.split('.')[2:])] = value

    return encoder_dict


class PredictorTrainer:
    def __init__(self, args):
        self.model_dir = args.model_dir
        self.device = (torch.device("cuda", 0))

        self.batch_size = args.batch_size
        self.num_worker = 4
        self.lr = 1e-3
        self.weight_decay = None
        self.num_epoch = 150
        self.predictor = Predictor(input_channel=args.num_channel, use_rgb=args.use_rgb, use_depth=args.use_depth,
                                   no_mask=args.no_mask, limited_fov=args.limited_fov,
                                   mean_pool_visual=args.mean_pool_visual, use_rgbd=args.use_rgbd,
                                   ).to(device=self.device, dtype=torch.float)
        self.dataloaders = dict()
        self.num_channel = args.num_channel

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.predictor.parameters()),
                                          weight_decay=args.weight_decay, lr=args.lr)

        if args.resume:
            # find the latest checkpoint in model-dir
            models_paths = list(
                filter(os.path.isfile, glob.glob(args.model_dir + "/ckpt_*.pth"))
            )
            models_paths.sort(key=os.path.getmtime)
            latest_ckpt_path = models_paths[-1]
            ckpt_dict = torch.load(latest_ckpt_path, map_location=self.device)
            self.predictor.load_state_dict(ckpt_dict['predictor'])
            logging.info(f'Loading the latest checkpoint: {latest_ckpt_path}')
            self.start_epoch = int(latest_ckpt_path.split('/')[-1].split('_')[-1].split('.')[0]) + 1
            if "optimizer" in ckpt_dict:
                self.optimizer.load_state_dict(ckpt_dict["optimizer"])
        else:
            self.start_epoch = 0

        print(f"Total parameters: {count_parameters(self.predictor)}")

    def run(self, splits, args, writer=None):
        datasets = dict()
        dataset_sizes = dict()
        for split in splits:
            datasets[split] = SoundSpacesSpeechDataset(split=split, use_real_imag=args.use_real_imag,
                                                       use_rgb=args.use_rgb, use_depth=args.use_depth,
                                                       limited_fov=args.limited_fov, use_rgbd=args.use_rgbd)
            self.dataloaders[split] = DataLoader(dataset=datasets[split],
                                                 batch_size=self.batch_size,
                                                 shuffle=(split == 'train'),
                                                 pin_memory=True,
                                                 num_workers=self.num_worker,
                                                 worker_init_fn=seed_worker
                                                 )

            dataset_sizes[split] = len(datasets[split])
            print('{} has {} samples'.format(split.upper(), dataset_sizes[split]))

        regressor_criterion = nn.MSELoss().to(device=self.device)
        if args.use_triplet_loss:
            triplet_criterion = nn.TripletMarginLoss(margin=args.triplet_margin).to(device=self.device)
        model = self.predictor
        metrics = speechmetrics.load(['stoi'])
        if args.step_decay:
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 90, 0.1)
        if args.exp_decay:
            # decay lr to 0.1 in 150 epochs
            scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, pow(0.1, 1 / 150))
        if args.linear_decay:
            # decay the learning rate linearly to 0 in 150 epochs
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                          lambda x: 1 - (x / float(self.num_epoch)))

        # training params
        since = time.time()
        best_split_loss = {}
        for split in splits:
            if split != 'train':
                best_split_loss[split] = float('inf')
        num_epoch = self.num_epoch if 'train' in splits else 1
        for epoch in range(self.start_epoch, num_epoch):
            logging.info('-' * 10)
            logging.info('Epoch {}/{}'.format(epoch, num_epoch))

            # Each epoch has a training and validation phase
            for split in splits:
                if split == 'train':
                    self.predictor.train()  # Set model to training mode
                else:
                    self.predictor.eval()  # Set model to evaluate mode

                running_total_loss = 0.0
                running_mag_loss = 0.0
                running_phase_loss = 0.0
                running_triplet_loss = 0.0
                running_metrics = defaultdict(float)
                num_data_point = 0
                num_metric_point = 0

                # Iterating over data once is one epoch
                for i, data in enumerate(tqdm(self.dataloaders[split])):
                    for key, value in data.items():
                        data[key] = value.to(device=self.device, dtype=torch.float)
                    receiver_spec = data['recv_spec']
                    source_spec = data['src_spec']

                    if args.log_mag:
                        assert not args.use_real_imag
                        if args.log1p:
                            receiver_spec[..., 0] = torch.log1p(receiver_spec[..., 0])
                            source_spec[..., 0] = torch.log1p(source_spec[..., 0])
                        else:
                            receiver_spec[..., 0] = torch.log10(receiver_spec[..., 0] + EPS)
                            source_spec[..., 0] = torch.log10(source_spec[..., 0] + EPS)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    cropped_receiver_spec = receiver_spec[:, 1:257, :256, :self.num_channel].permute(0, 3, 1, 2)
                    cropped_source_spec = source_spec[:, 1:257, :256, :self.num_channel].permute(0, 3, 1, 2)
                    data.update({'spectrograms': cropped_receiver_spec})
                    if split == 'train':
                        output = model(data)
                    else:
                        with torch.no_grad():
                            output = model(data)

                    pred_mask = output['pred_mask']
                    if args.no_mask:
                        pred_spec = pred_mask
                    else:
                        if self.num_channel == 1:
                            pred_spec = pred_mask * cropped_receiver_spec
                        else:
                            if args.use_real_imag:
                                pred_spec_real = cropped_receiver_spec[:, 0, :, :] * pred_mask[:, 0, :, :] \
                                                 - cropped_receiver_spec[:, 1, :, :] * pred_mask[:, 1, :, :]
                                pred_spec_img = cropped_receiver_spec[:, 0, :, :] * pred_mask[:, 1, :, :] \
                                                + cropped_receiver_spec[:, 1, :, :] * pred_mask[:, 0, :, :]
                                pred_spec = torch.cat((pred_spec_real.unsqueeze(1), pred_spec_img.unsqueeze(1)), 1)
                            else:
                                pred_spec_mag = cropped_receiver_spec[:, 0, :, :] * pred_mask[:, 0, :, :]
                                pred_spec_phase = cropped_receiver_spec[:, 1, :, :] * pred_mask[:, 1, :, :]
                                pred_spec = torch.cat((pred_spec_mag.unsqueeze(1), pred_spec_phase.unsqueeze(1)), 1)

                    mag_loss = regressor_criterion(pred_spec[:, 0, :, :], cropped_source_spec[:, 0, :, :])

                    if args.phase_loss == 'sin':
                        pred_spec_phase_2d = torch.cat((torch.sin(pred_spec[:, 1, :, :]).unsqueeze(1),
                                                        torch.cos(pred_spec[:, 1, :, :]).unsqueeze(1)), 1)
                        cropped_source_spec_phase_2d = torch.cat(
                            (torch.sin(cropped_source_spec[:, 1, :, :]).unsqueeze(1),
                             torch.cos(cropped_source_spec[:, 1, :, :]).unsqueeze(1)), 1)
                        phase_loss = regressor_criterion(pred_spec_phase_2d, cropped_source_spec_phase_2d)
                    elif args.phase_loss == 'cos':
                        phase_loss = -F.cosine_similarity(pred_spec[:, 1, :, :].reshape((pred_spec.shape[0], -1)),
                                                          cropped_source_spec[:, 1, :, :].reshape(
                                                              (pred_spec.shape[0], -1))).mean()
                    else:
                        phase_loss = regressor_criterion(pred_spec[:, 1, :, :], cropped_source_spec[:, 1, :, :])

                    if args.use_triplet_loss:
                        # use a random feature offset by 1 as the negative sample
                        ind_neg = torch.tensor((np.arange(data['recv_spec'].size(0)) + 1) % data['recv_spec'].size(0),
                                               device=self.device)
                        triplet_loss = triplet_criterion(output['visual_feat'], output['audio_feat'],
                                                         output['visual_feat'][ind_neg])

                    loss = mag_loss + phase_loss * args.phase_weight + \
                           (triplet_loss * args.triplet_loss_weight if args.use_triplet_loss else 0)

                    # backward + optimize only if in training phase
                    if split == 'train':
                        loss.backward()
                        self.optimizer.step()

                    running_total_loss += loss.item() * source_spec.shape[0]
                    running_mag_loss += mag_loss.item() * source_spec.shape[0]
                    running_phase_loss += phase_loss.item() * source_spec.shape[0]
                    if args.use_triplet_loss:
                        running_triplet_loss += triplet_loss.item() * source_spec.shape[0]
                    num_data_point += source_spec.shape[0]

                    if args.reconstruct_audio:
                        num_sample = self.batch_size if split == 'test' else 1
                        full_pred_spec = receiver_spec.clone()
                        full_pred_spec[:, 1:257, :256, :self.num_channel] = pred_spec.permute(0, 2, 3, 1).detach()
                        if self.num_channel == 1:
                            mag = torch.pow(10, full_pred_spec[:num_sample, :, :, 0])
                            phase = full_pred_spec[:num_sample, :, :, 1]
                            predicted_audio = griffinlim(mag, phase, torch.hamming_window(400, device=self.device),
                                                         n_fft=512, hop_length=160, win_length=400, power=1, n_iter=0)
                        else:
                            predicted_audio = torch.istft(full_pred_spec[:num_sample], n_fft=512, hop_length=160,
                                                          win_length=400,
                                                          window=torch.hamming_window(400, device=self.device))

                        predicted_audio = predicted_audio.cpu().numpy()
                        source_audio = source_audio.cpu().numpy()[:num_sample]

                        if args.save_audio:
                            audio_dir = os.path.join(self.model_dir, 'generated_audio')
                            os.makedirs(audio_dir, exist_ok=True)
                        for k in range(source_audio.shape[0]):
                            running_metrics['stoi'] += metrics(predicted_audio[k], source_audio[k], rate=16000)['stoi'][
                                0]
                            running_metrics['pesq'] += pesq(16000, source_audio[k], predicted_audio[k], 'wb')

                            if args.save_audio:
                                # only generate samples for the first batch
                                file_path = os.path.join(audio_dir, f'{k}.wav')
                                librosa.output.write_wav(file_path, predicted_audio[k], 16000)

                        num_metric_point += num_sample

                epoch_total_loss = running_total_loss / num_data_point
                if writer is not None:
                    writer.add_scalar(f'{split}/total_loss', epoch_total_loss, epoch)
                    writer.add_scalar(f'{split}/mag_loss', running_mag_loss / num_data_point, epoch)
                    writer.add_scalar(f'{split}/phase_loss', running_phase_loss / num_data_point, epoch)
                    if args.use_triplet_loss:
                        writer.add_scalar(f'{split}/triplet_loss', running_triplet_loss / num_data_point, epoch)
                    for metric in running_metrics:
                        writer.add_scalar(f'{split}/{metric}', running_metrics[metric] / num_metric_point, epoch)
                logging.info(f'{split.upper()} Total loss: {epoch_total_loss:.4f}')
                for metric in running_metrics:
                    logging.info(f'{split.upper()} {metric}: {running_metrics[metric] / num_metric_point:.4f}')

                if split == 'train' and epoch % args.save_ckpt_interval == 0:
                    self.save_checkpoint(f"ckpt_{epoch}.pth", args)

                if split in best_split_loss and epoch_total_loss < best_split_loss[split]:
                    best_split_loss[split] = epoch_total_loss
                    self.save_checkpoint(f"best_{split}.pth", args)

            if args.step_decay or args.exp_decay or args.linear_decay:
                scheduler.step()

        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        for split in best_split_loss:
            logging.info(f'Best {split} loss: {best_split_loss[split]:4f}')

    def save_checkpoint(self, ckpt_path, args, checkpoint=None):
        if checkpoint is None:
            checkpoint = {
                "predictor": self.predictor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "args": args
            }
        torch.save(
            checkpoint, os.path.join(self.model_dir, ckpt_path)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-type", choices=["train", "eval"], default='train')
    parser.add_argument("--model-dir", default='data/models/dereverb/audio_visual')
    parser.add_argument("--eval-ckpt", default='')
    parser.add_argument("--eval-best", default=False, action='store_true')
    parser.add_argument("--overwrite", default=False, action='store_true')
    parser.add_argument("--use-rgb", default=False, action='store_true')
    parser.add_argument("--use-depth", default=False, action='store_true')
    parser.add_argument("--use-rgbd", default=False, action='store_true')
    parser.add_argument("--num-channel", default=1, type=int)
    parser.add_argument("--batch-size", default=96, type=int)
    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--log-mag", default=False, action='store_true')
    parser.add_argument("--use-real-imag", default=False, action='store_true')
    parser.add_argument("--no-mask", default=False, action='store_true')
    parser.add_argument("--log1p", default=False, action='store_true')
    parser.add_argument("--reconstruct-audio", default=False, action='store_true')
    parser.add_argument("--save-audio", default=False, action='store_true')
    parser.add_argument("--phase-weight", default=0.1, type=float)
    parser.add_argument("--phase-loss", default='', type=str)
    parser.add_argument("--test-split", default='test-unseen', type=str)
    parser.add_argument("--save-ckpt-interval", default=1, type=int)
    parser.add_argument("--use-triplet-loss", default=False, action='store_true')
    parser.add_argument("--triplet-loss-weight", default=0.001, type=float)
    parser.add_argument("--triplet-margin", default=1, type=float)
    parser.add_argument("--limited-fov", default=False, action='store_true')
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--step-decay", default=False, action='store_true')
    parser.add_argument("--exp-decay", default=False, action='store_true')
    parser.add_argument("--linear-decay", default=False, action='store_true')
    parser.add_argument("--mean-pool-visual", default=False, action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    if args.eval_best:
        assert args.run_type == 'eval'
    if args.run_type == 'train' and os.path.exists(args.model_dir):
        overwrite = args.overwrite or (input('Model dir exists. Overwrite?\n') == 'y')
        if overwrite:
            shutil.rmtree(args.model_dir)

    run(args)


def run(args):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    predictor_trainer = PredictorTrainer(args)

    if args.run_type == 'train':
        writer = SummaryWriter(log_dir=os.path.join(args.model_dir, 'tb'))
        predictor_trainer.run(['train', 'val'], args, writer=writer)
    else:
        if args.eval_best:
            ckpt = torch.load(os.path.join(args.model_dir, 'best_val.pth'))
        else:
            ckpt = torch.load(args.eval_ckpt)
        predictor_trainer.predictor.load_state_dict(ckpt['predictor'])
        
        # only evaluates the model on the loss
        predictor_trainer.run([args.test_split], args)


if __name__ == '__main__':
    main()
