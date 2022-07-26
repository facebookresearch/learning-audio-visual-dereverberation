# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup


setup(
    name='learning-audio-visual-dereverberation',
    version='0.1.0',
    packages=[
        'vida',
    ],
    install_requires=[
        'speechbrain',
        'torch',
        'numpy>=1.16.1',
        'yacs>=0.1.5',
        'attrs>=19.1.0',
        'imageio>=2.2.0',
        'imageio-ffmpeg>=0.2.0',
        'scipy>=1.0.0',
        'tqdm>=4.0.0',
        'Pillow',
        'getch',
        'matplotlib',
        'librosa',
        'torchsummary',
        'gitpython',
        'tqdm',
        'notebook',
        'astropy',
        'scikit-image',
        'speechmetrics @ git+https://github.com/aliutkus/speechmetrics',
        'pesq',
        'tensorboard',
        'torchaudio',
        'torchvision'
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
