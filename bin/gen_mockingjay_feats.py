#!/usr/bin/env python3
"""Extract acoustic features using mockingjay model.

To extract features using a pretrained mockingjay model:

    python gen_mockingjay_feats.py --use_gpu state-500000.pt feats_dir/ \
        fn1.wav fn2.wav fn3.wav ...

which will load a pretrained model from the checkpoint located at

     state-500000.pt

apply it to the audio files ``fn1.wav, ``fn2.wav``, ``fn3.wav``, ..., and for
each recording output frame-level features to a corresponding ``.npy`` file
located under the directory ``feats_dir``. The flag ``--use_gpu`` instructs
the script to use the GPU, if free.

For each audio file, this script outputs a NumPy ``.npy`` file containing an
``num_frames`` x ``num_features`` array of frame-level features. These
correspond to frames sampled every 12.5 ms starting from an offset of 0; that
is, the ``i``-th frame of features corresponds to an offset of ``i*0.0125``
seconds.
"""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformer.nn_transformer import TRANSFORMER
from utility.audio import extract_feature


def get_device(x):
    """Return device that instance of ``Module`` or ``Tensor`` is on."""
    if isinstance(x, torch.Tensor):
        return x.device
    elif isinstance(x, torch.nn.Module):
        return next(x.parameters()).device
    else:
        raise ValueError(f'"x" must be an instance of Module or Tensor, not '
                         f'{type(x)}')


def extract_feats_to_file(npy_path, audio_path, mockingjay_model):
    """Extract features to file.

    Parameters
    ----------
    npy_path : Path
        Path to output ``.npy`` file.

    audio_path : Path
        Path to audio file to extract features for.

    mockingjay_model : transformer.nn_transformer.TRANSFORMER
        mackingjay model.
    """
    dev = get_device(mockingjay_model)

    # Pre-process
    x = extract_feature(audio_path, feature='mel', delta=True)
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to(dev)

    # Extract context layer features from mockingjay model.
    with torch.no_grad():
        feats = mockingjay_model(x)

    # Save as uncompressed .npy file.
    feats = feats.cpu().numpy()  # Shape: 1 x feat_dim x n_frames
    feats = np.squeeze(feats)
    np.save(npy_path, feats)


def main():
    parser = argparse.ArgumentParser(
        description='generate mockingjay features', add_help=True)
    parser.add_argument(
        '--use_gpu', default=False, action='store_true', help='use GPU')
    parser.add_argument(
        'modelf', type=Path, help='mockingjay checkpoint')
    parser.add_argument(
        'feats_dir', type=Path,
        help='path to output directory for .npy files')
    parser.add_argument(
        'afs', nargs='*', type=Path, help='audio files to be processed')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    os.makedirs(args.feats_dir, exist_ok=True)

    # Determine device for computation.
    use_gpu = args.use_gpu and torch.cuda.is_available()
    device = 'cuda:0' if use_gpu else 'cpu'
    device = torch.device(device)

    options = {
        'ckpt_file'     : args.modelf,
        'load_pretrain' : 'True',
        'no_grad'       : 'True',
        'dropout'       : 'default',
        'spec_aug'      : 'False',
        'spec_aug_prev' : 'True',
        'weighted_sum'  : 'False',
        'select_layer'  : -1,
        'permute_input' : 'False',
        # Set to False to take input as (B, T, D), otherwise take (T, B, D)
    }

    # setup the transformer model
    # copying a param with shape torch.Size([768, 480]) from checkpoint
    # set inp_dim to 0 for auto setup
    mockingjay_model = TRANSFORMER(options=options, inp_dim=0)
    mockingjay_model = mockingjay_model.to(device)

    # Process.
    with tqdm(total=len(args.afs)) as pbar:
        for fn in args.afs:
            npy_path = Path(args.feats_dir, fn.stem + '.npy')
            try:
                extract_feats_to_file(npy_path, fn, mockingjay_model)
            except RuntimeError:
                tqdm.write(f'ERROR: CUDA OOM error when processing "{fn}". \
                             Skipping.')
                torch.cuda.empty_cache()
            pbar.update(1)


if __name__ == '__main__':
    main()
