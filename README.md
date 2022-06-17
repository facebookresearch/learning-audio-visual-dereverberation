# Learning Audio-Visual Dereverberation

## Motivation
Reverberation from audio reflecting off surfaces and objects in the environment not only degrades the quality of speech for human perception, but also severely impacts the accuracy of automatic speech recognition. Prior work attempts to remove reverberation based on the audio modality only. Our idea is to learn to dereverberate speech from audio-visual observations. The visual environment surrounding a human speaker reveals important cues about the room geometry, materials, and speaker location, all of which influence the precise reverberation effects in the audio stream. We introduce Visually-Informed Dereverberation of Audio (VIDA), an end-to-end approach that learns to remove reverberation based on both the observed sounds and visual scene. In support of this new task, we develop a large-scale dataset that uses realistic acoustic renderings of speech in real-world 3D scans of homes offering a variety of room acoustics. Demonstrating our approach on both simulated and real imagery for speech enhancement, speech recognition, and speaker identification, we show it achieves state-of-the-art performance and substantially improves over traditional audio-only methods.

## Citation
If you find this paper and code useful, please cite the following [paper](https://arxiv.org/pdf/2106.07732.pdf):
```
@arxiv{chen22av_dereverb,
  title     =     {Learning Audio-Visual Dereverberation,
  author    =     {Changan Chen and Wei Sun and David Harwath and Kristen Grauman},
  journal   =     {arXiv},
  year      =     {2022}
}
```

## Installation 
Install this repo into pip by running the following command:
```
pip install -e .
```

## Usage
1. Training
```angular2html
py vida/trainer.py --model-dir data/models/vida  --num-channel 2 --use-depth --use-rgb --log-mag --no-mask --phase-loss sin --phase-weight 0.1 --use-triplet-loss --exp-decay --triplet-margin 0.5 --mean-pool-visual --overwrite
```
2. Evaluation (instructions coming)


## Data
See the [data page](vida/data.md) for instructions on how to download the data


## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
This repo is CC-BY-NC licensed, as found in the [LICENSE](LICENSE) file.