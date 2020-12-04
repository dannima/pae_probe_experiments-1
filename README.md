## Overview
This repository is the implementation of probing experiments on various recent pre-trained acoustic models: wav2vec, vq-wav2vec, Mockingjay, DeCoAR; as well as traditional MFCC and Mel scale filterbank features.

## Prerequisites
* **Bash 4** (or above)
* **Python 3**
* Required packages can be installed by running `pip install -r requirements.txt`. If you are not using GPU with CUDA 10.1, you might want to change `torch==1.7.0+cu101` and `mxnet_cu101mkl` in `requirements.txt` to your own CUDA version before installation.

## Reproducing Experiments
1. Open `path.sh`, change `*TIMIT_PATH` to the path that your dataset is stored. Open `run_probe_exp.sh`, change `NUM_GPU` to the number of GPUs you have.

2. Each subdirectory in `models/` contains the code for running probing experiments using one kind of acoustic features. Enter `models/${FEATS}/`, follow the instructions in `README.md` (e.g. `models/mockingjay/README.md`). If there is no `README.md` file, simply run `./run.sh` to reproduce the results.

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