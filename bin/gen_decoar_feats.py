#!/scratch/dannima/mockingjay/bin/python
"""Extract acoustic features using DeCoAR model.

To extract features using a pretrained DeCoAR model:

    python gen_decoar_feats.py --use_gpu \
    artifacts/decoar-encoder-29b8e2ac.params feats_dir/ \
        fn1.wav fn2.wav fn3.wav ...

which will load a pretrained model located at

    artifacts/decoar-encoder-29b8e2ac.params

apply it to the audio files ``fn1.wav, ``fn2.wav``, ``fn3.wav``, ..., and for
each recording output frame-level features to a corresponding ``.npy`` file
located under the directory ``feats_dir``. The flag ``--use_gpu`` instructs
the script to use the GPU, if free.

For each audio file, this script outputs a NumPy ``.npy`` file containing an
``num_frames`` x ``num_features`` array of frame-level features. These
correspond to frames sampled every 10 ms starting from an offset of 0; that is,
the ``i``-th frame of features corresponds to an offset of ``i*0.010`` seconds.
"""

import os
import sys
import argparse
import mxnet as mx
import numpy as np
from tqdm import tqdm
from pathlib import Path
from speech_reps.featurize import DeCoARFeaturizer


def extract_feats_to_file(npy_path, audio_path, featurizer):
    """Extract features to file.

    Parameters
    ----------
    npy_path : Path
        Path to output ``.npy`` file.

    audio_path : Path
        Path to audio file to extract features for.
    """
    # Returns a (time, feature) NumPy array
    data = featurizer.file_to_feats(audio_path)
    np.save(npy_path, data)


def main():
    parser = argparse.ArgumentParser(
        description='generate DeCoAR features', add_help=True)
    parser.add_argument(
        '--use_gpu', default=False, action='store_true', help='use GPU')
    parser.add_argument(
        'modelf', type=Path, help='DeCoAR pre-trained model')
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
    use_gpu = args.use_gpu and mx.test_utils.list_gpus()
    device = 0 if use_gpu else None

    # Load the model on device
    featurizer = DeCoARFeaturizer(args.modelf, gpu=device)

    # Process.
    with tqdm(total=len(args.afs)) as pbar:
        for fn in args.afs:
            npy_path = Path(args.feats_dir, fn.stem + '.npy')
            try:
                extract_feats_to_file(npy_path, fn, featurizer)
            except RuntimeError:
                tqdm.write(f'ERROR: CUDA OOM error when processing "{fn}". \
                             Skipping.')
            pbar.update(1)


if __name__ == '__main__':
    main()
