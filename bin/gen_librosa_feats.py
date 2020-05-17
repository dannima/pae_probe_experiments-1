#!/usr/bin/env python3
import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import sys
import warnings
import yaml

import librosa
from librosa.feature import melspectrogram, mfcc
import numpy as np
from tqdm import tqdm


warnings.simplefilter('ignore', FutureWarning)


def sec_to_samples(n_sec, sr):
    """Return number of samples required to cover duration in seconds."""
    return int(n_sec*sr)


def db_mel_spectrogram(x, sr, **kwargs):
    """Return DB mel spectrogram."""
    p = melspectrogram(x, sr, **kwargs)
    p = librosa.power_to_db(p, ref=np.max)
    return p


def extract_feats(af, feats_dir, sr, feats_fn, add_deltas=False, **kwargs):
    """Extract mel frequency cepstral coefficient features to file.

    Parameters
    ----------
    af : Path
        Path to audio file.

    feats_dir : Path
        Path to output directory.

    sr : int
       Sample rate (Hz) to resample audio to.

    kwargs
        Keyword arguments to pass to ``librosa.feature.melspectrogram``.
    """
    x, _ = librosa.core.load(af, sr)
    feats = feats_fn(x, sr=sr, **kwargs)
    feats = feats.T # Convert features to frames x feat_dim.
    if add_deltas:
        delta = librosa.feature.delta(feats, order=1, axis=0)
        delta_delta = librosa.feature.delta(feats, order=2, axis=0)
        feats = np.concatente(
            [feats, delta, delta_delta], axis=1)
    np.save(Path(feats_dir, af.stem + '.npy'), feats)


def main():
    parser = argparse.ArgumentParser(
        'extract librosa spectral features from audio', add_help=True)
    parser.add_argument(
        'feats_dir', nargs=None, type=Path, 
        help='output directory for features')
    parser.add_argument(
        'af', nargs='+', type=Path,
        help='audio files to process')
    parser.add_argument(
        '--ftype', nargs=None, default='mfcc', metavar='FEATURE',
        choices=['mfcc', 'mel_fbank'],
        help='type of features to extract (default: %(default)s)')
    parser.add_argument(
        '--step', nargs=None, default=0.01, type=float, metavar='SECONDS',
        help='frame step in seconds (default: %(default)s)')
    parser.add_argument(
        '--wl', nargs=None, default=0.035, type=float, metavar='SECONDS',
        help='window length in seconds (default: %(default)s)')
    parser.add_argument(
        '--sr', nargs=None, default=16000, type=int, metavar='SAMPLES',
        help='resample to SAMPLES Hz before extraction (default: %(default)s)')
    parser.add_argument(
        '--add_deltas', default=False, action='store_true',
        help='compute delta and delta-delta features')
    parser.add_argument(
        '--config', nargs=None, default=None, type=Path, metavar='FILE',
        help='load additional keyword arguments for librosa feature function '
             'from YAML config FILE (default: %(default)s)')
    parser.add_argument(
        '--n_jobs', nargs=None, default=1, type=int, metavar='JOBS',
        help='number of parallel jobs (default: %(default)s)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    args.feats_dir.mkdir(parents=True, exist_ok=True)

    # Determine parameters for MFCC extraction.
    kwargs = {}
    kwargs['win_length'] = sec_to_samples(args.wl, args.sr)
    kwargs['hop_length'] = sec_to_samples(args.step, args.sr)
    if args.config is not None:
        with open(args.config, 'r') as f:
            kwargs.update(yaml.load(f))
    if 'n_fft' in kwargs:
        # Make sure n_fft is at least as large as the window size.
        kwargs['n_fft'] = int(2**np.ceil(np.log2(kwargs['win_length'])))

    # Extract features in parallel.
    if args.ftype == 'mfcc':
        feats_fn = mfcc
    elif args.ftype == 'mel_fbank':
        feats_fn = db_mel_spectrogram
    audio_paths = sorted(args.af)
    pool = Pool(args.n_jobs)
    f = partial(
        extract_feats, feats_dir=args.feats_dir, sr=args.sr, feats_fn=feats_fn,
        **kwargs)
    with tqdm(total=len(audio_paths)) as pbar:
        for _ in pool.imap(f, audio_paths):
            pbar.update(1)


if __name__ == '__main__':
    main()
