# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
from collections import defaultdict
import glob
from itertools import combinations, product
import os
import pickle
import random
from tqdm import tqdm

import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def overlap_chunk(input, dimension, size, step, left_padding):
    """
    Input shape is [Frequency bins, Frame numbers]
    """
    input = F.pad(input, (left_padding, size), 'constant', 0)
    return input.unfold(dimension, size, step)

# Prepare the meta data
def parse_csv(data_list):
    test_list = {}
    with open(data_list) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        count = 0
        for row in csv_reader:
            if count > 0:
                id, dur, path, target = row[0], row[1], row[2], row[4].strip()
                test_list[id] = (path, target, float(dur))
            count += 1
    return test_list

def compute_scores(pairs, verification, enhance_cache):
    size = len(pairs)
    batch = 16
    scores = []
    for i in tqdm(range(0, size, batch)):
        s1, s2 = [], []
        wav_len1, wav_len2 = [], []
        for pair in pairs[i:i + batch]:
            key_1, key_2 = pair
            s1.append(enhance_cache[key_1][0].cuda().squeeze(0))
            s2.append(enhance_cache[key_2][1].cuda().squeeze(0))
            wav_len1.append(s1[-1].size(0))
            wav_len2.append(s2[-1].size(0))

        wav_len1, wav_len2 = torch.tensor(wav_len1).cuda().float(), torch.tensor(wav_len2).cuda().float()
        wav_len1, wav_len2 = wav_len1 / torch.max(wav_len1), wav_len2 / torch.max(wav_len2)

        ref = pad_sequence(s1, batch_first=True)
        compare = pad_sequence(s2, batch_first=True)
        score, prediction = verification.verify_batch(ref, compare, wav_len1, wav_len2)
        scores.append(score)
    return torch.cat(scores).squeeze()

def load_spkcfg(files, split):
    spkrec_config = os.path.join('data', split, 'spkrec_config.ark')

    if os.path.exists(spkrec_config):
        with open(spkrec_config, 'rb') as fo:
            data = pickle.load(fo)
            print(f"loading...{spkrec_config}")
        return data['pos'], data['neg']
    else:
        with open("data/sounds/speech/LibriSpeech/SPEAKERS.TXT") as f:
            content = f.readlines()
        id2sex = {}
        for line in content[12:]:
            detail = line.split('|')
            id = detail[0].strip()
            sex = detail[1].strip()
            id2sex[id] = sex

        spkid2file = defaultdict(list)
        non_duplicate_speech_ids = []
        file2sex = {}
        for file in files:
            spkid = file.split('/')[-1].split('-')[0]
            scene_id = file.split('/')[-2]
            speech_id = file.split('/')[-1].split('.')[0]
            non_duplicate_speech_id = f'{speech_id}_{scene_id}'
            spkid2file[spkid].append(non_duplicate_speech_id)
            non_duplicate_speech_ids.append(non_duplicate_speech_id)
            file2sex[non_duplicate_speech_id] = id2sex[spkid]
        pos_pairs, neg_pairs = list(), list()
        for k, v in spkid2file.items():
            pos_pairs += combinations(v, r=2)
            neg_pairs += product(v, [x for x in non_duplicate_speech_ids if not x.startswith(k)])
        num_sample = 1000 if split == 'val-mini' else 40000
        random.shuffle(pos_pairs)
        random.shuffle(neg_pairs)
        pos_pairs = pos_pairs[:num_sample]
        selected_neg_pairs = []
        count = 0
        for pair in neg_pairs:
            speech_0, speech_1 = pair
            if file2sex[speech_0] != file2sex[speech_1]:
                count += 1
                selected_neg_pairs.append(pair)
            if count >= num_sample:
                break
        with open(spkrec_config, 'wb') as fo:
            pickle.dump({'pos': pos_pairs, 'neg': selected_neg_pairs}, fo)
        return pos_pairs, selected_neg_pairs

def load_noise(split):
    splitmap = {"train": "tr", "val": "cv", "val-mini": "cv", "test-unseen": "tt", "test-seen": "tt"}
    noise_cache_path = os.path.join('data/sounds/speech/wham_noise', splitmap[split], "noise.pkl")
    if os.path.exists(noise_cache_path):
        with open(noise_cache_path, 'rb') as fo:
            data = pickle.load(fo)
        noise_pool = data['noise']
    else:
        noise_files = sorted(glob.glob(os.path.join('data/sounds/speech/wham_noise', splitmap[split], "*.wav")))
        noise_files = np.random.choice(noise_files, 600)
        noise_list = []
        for noise_file in tqdm(noise_files):
            noise, _ = torchaudio.load(noise_file)
            noise_list.append(noise)
        noise_pool = torch.cat(noise_list, dim=1)[0, :]
        with open(noise_cache_path, 'wb') as fo:
            pickle.dump({'noise': noise_pool}, fo)
    return noise_pool
