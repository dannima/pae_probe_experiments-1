## Overview
This repository is the implementation of probing experiments on various recent pre-trained acoustic models:

* [wav2vec](https://arxiv.org/pdf/1904.05862.pdf)
* [vq-wav2vec](https://arxiv.org/pdf/1910.05453.pdf)
* [Mockingjay](https://arxiv.org/pdf/1910.12638.pdf)
* [DeCoAR](https://arxiv.org/pdf/1912.01679.pdf)
* [wav2vec2.0](https://arxiv.org/pdf/2006.11477.pdf)

as well as traditional MFCC and Mel scale filterbank features.

## Prerequisites
* **Bash** version &ge; 4
* **Python** version &ge; 3.6
* If you are _not_ using GPU with CUDA 10.1, you might want to change `torch==1.7.0+cu101` and `mxnet_cu101mkl` in `requirements.txt` to your own CUDA version before installation.
* Required packages can be installed by running `./install.sh`.

## Reproducing Experiments
1. Open `path.sh`, change `*TIMIT_PATH` to the path that your dataset is stored. Open `run_probe_exp.sh`, change `NUM_GPU` to the number of GPUs you have.

2. Each subdirectory in `models/` contains the code for running probing experiments using one kind of acoustic features. Enter `models/${FEATS}/`, follow the instructions in `README.md` (e.g. `models/mockingjay/README.md`). If there is no `README.md` file, simply run `./run.sh` to reproduce the results.

	### How `run.sh` works:
	Under `models/${FEATS}/`, `run.sh` will first extract acoustic features and store them in `feats/`. If `$FEATS` is a pre-trained acoustic representation, the model checkpoints and related files will be downloaded in `checkpoints/`. Then, `run.sh` will generate configuration files for diffrent tasks, and store them in `configs/tasks/`. Finally, it calls `run_probe_exp.sh` in its grandparent directory to run probing experiments. The results are stored in `logs/`.
	
## References
This is the code repository for the paper [Probing Acoustic Representations for Phonetic Properties](https://arxiv.org/pdf/2010.13007.pdf). If you use the code in your work, please cite

```
@article{ma2020probing,
  title={Probing Acoustic Representations for Phonetic Properties},
  author={Ma, Danni and Ryant, Neville and Liberman, Mark},
  journal={arXiv preprint arXiv:2010.13007},
  year={2020}
}
```
