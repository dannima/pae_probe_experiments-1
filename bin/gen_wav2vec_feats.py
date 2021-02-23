#!/usr/bin/env python3
"""Extract acoustic features using wav2vec, vq-wav2vec or wav2vec 2.0 models.

To extract features using a pretrained wav2vec model:

    python gen_wav2vec_feats.py --use_gpu w2v_model.pt feats_dir/ \
        fn1.wav fn2.wav fn3.wav ...

which will load a pretrained model from the checkpoint located at

     w2v_model.pt

apply it to the audio files ``fn1.wav, ``fn2.wav``, ``fn3.wav``, ..., and for
each recording output frame-level features to a corresponding ``.npy`` file
located under the directory ``feats_dir``. The flag ``--use_gpu`` instructs
the script to use the GPU, if free.

For each audio file, this script outputs a NumPy ``.npy`` file containing an
``num_frames`` x ``num_features`` array of frame-level features. These
correspond to frames sampled every 10 ms starting from an offset of 0; that is,
the ``i``-th frame of features corresponds to an offset of ``i*0.010`` seconds.

If you would instead like to extract RoBERTa features from a RoBERTa model
processing quantized features output by a vq-wav2ec model, run:

    python gen_wav2vec_feats.py --use_gpu \
        --roberta roberta_model.pt --vocab dict.txt \
        vqw2v_model.pt feats_dir/ fn1.wav fn2.wav fn3.wav ...

where ``vqw2v_model.pt`` is the checkpoint from a pretrained vq-wav2vec model,
``roberta_model.pt`` is the corresponding pretrained RoBERTa model checkpoint,
and ``dict.txt`` is the vocabulary used by that RoBERTa model.

If you would like to extract wav2vec 2.0 features from a wav2vec 2.0 model,
run:

    python gen_wav2vec_feats.py --use_gpu --v2 w2v2_model.pt feats_dir/ \
        fn1.wav fn2.wav fn3.wav ...
where ``w2v2_model.pt`` is the checkpoint from a pretrained wav2vec 2.0 model.
"""
import argparse
import os
from pathlib import Path
import sys
import fairseq
from fairseq.data import Dictionary
from fairseq.models.roberta import RobertaModel
from fairseq.models.wav2vec import Wav2VecModel
from fairseq.tasks.masked_lm import MaskedLMTask
import librosa
import numpy as np
import torch
from tqdm import tqdm
from wurlitzer import pipes


activation = {}


def get_activation(name):
    """Extract output of an intermediate layer"""
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def get_device(x):
    """Return device that instance of ``Module`` or ``Tensor`` is on."""
    if isinstance(x, torch.Tensor):
        return x.device
    elif isinstance(x, torch.nn.Module):
        return next(x.parameters()).device
    else:
        raise ValueError(f'"x" must be an instance of Module or Tensor, not '
                         f'{type(x)}')


def extract_feats_to_file(npy_path, audio_path, wav2vec_model,
                          roberta_model=None, v2=False):
    """Extract features to file.

    Parameters
    ----------
    npy_path : Path
        Path to output ``.npy`` file.

    audio_path : Path
        Path to audio file to extract features for.

    wav2vec_model : fairseq.models.Wav2VecModel
        Wav2vec model.

    roberta_model : fairseq.models.RobertaModel, optional
        Roberta model.
        (Parameters: None)
    """
    dev = get_device(wav2vec_model)

    # Convert to 16 kHz.
    x, sr = librosa.load(audio_path, sr=16000)
    x = torch.from_numpy(x).unsqueeze(0)
    x = x.to(dev)

    # Extract context layer features from wav2vec model.
    if not v2:
        with torch.no_grad():
            z = wav2vec_model.feature_extractor(x)
            c = wav2vec_model.feature_aggregator(z)
            feats = c

        # If a RoBERTa model is specified, perform quantization
        # and pass through RoBERTa to get features.
        # https://github.com/pytorch/fairseq/issues/1793
        if roberta_model is not None:
            _, idx = wav2vec_model.vector_quantizer.forward_idx(z)
            idx = idx.squeeze(0).cpu().numpy()
            tokens = [f'{g1}-{g2}' for g1, g2 in idx]
            sent = ' '.join(tokens)
            tokens = roberta_model.encoder.dictionary.encode_line(
                sent, append_eos=False, add_if_not_exist=False)
            tokens = tokens.long().unsqueeze(0)
            tokens = tokens.to(dev)
            with torch.no_grad():
                feats, _ = roberta_model.encoder.extract_features(tokens)
            feats = feats.T  # For some reason Roberta feats are inverted.

        # Save as uncompressed .npy file.
        feats = feats.cpu().numpy()  # Shape: feat_dim x n_frames x 1
        feats = np.squeeze(feats)
        feats = feats.T  # Shape: n_frames x feat_dim
        np.save(npy_path, feats)

    # If a wav2vec 2.0 model is specified
    else:
        with torch.no_grad():
            handle = \
                wav2vec_model.w2v_encoder.final_dropout.register_forward_hook(
                    get_activation('final_dropout'))
            _ = wav2vec_model(source=x, padding_mask=None)
            feats = activation['final_dropout']
            handle.remove()

            # feats = wav2vec_model.extract_features(
            #     source=x, padding_mask=None)['encoder_out']

        # Save as uncompressed .npy file.
        feats = feats.cpu().numpy()  # Shape: n_frames x 1 x feat_dim
        feats = np.squeeze(feats)  # Shape: n_frames x feat_dim
        np.save(npy_path, feats)


def main():
    parser = argparse.ArgumentParser(
        description='generate wav2vec/vq-wav2vec features', add_help=True)
    parser.add_argument(
        '--roberta', default=None, type=Path, metavar='STR',
        help='path to RoBERTa checkpoint (default: %(default)s)')
    parser.add_argument(
        '--vocab', default=None, type=Path, metavar='STR',
        help='path to vocabulary file (default: %(default)s)')
    parser.add_argument(
        '--v2', default=False, action='store_true',
        help='use wav2vec 2.0 model')
    parser.add_argument(
        '--use_gpu', default=False, action='store_true',
        help='use GPU')
    parser.add_argument(
        'modelf', type=Path, help='wav2vec/vq-wav2vec/wav2vec 2.0 checkpoint')
    parser.add_argument(
        'feats_dir', type=Path,
        help='path to output directory for .npy files')
    parser.add_argument(
        'afs', nargs='*', type=Path, help='audio files to be processed')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    if args.roberta and args.v2:
        parser.error('Can\'t load vq-wav2vec and wav2vec 2.0 model together!')

    os.makedirs(args.feats_dir, exist_ok=True)

    # Determine device for computation.
    use_gpu = args.use_gpu and torch.cuda.is_available()
    device = 'cuda:0' if use_gpu else 'cpu'
    device = torch.device(device)

    # Load RoBERTa model.
    using_roberta = args.roberta is not None and args.vocab is not None
    roberta_model = None
    if using_roberta:
        with open(args.vocab, 'r') as f:
            d = Dictionary.load(f)
        roberta_cp = torch.load(args.roberta)
        roberta_args = roberta_cp['args']
        roberta_cp['args'].quant_noise_pq = 0.0
        roberta_cp['args'].quant_noise_pq_block_size = 8
        roberta_model = RobertaModel.build_model(
            roberta_args, MaskedLMTask(roberta_args, d))
        roberta_model.load_state_dict(roberta_cp['model'])
        roberta_model.eval()
        roberta_model = roberta_model.to(device)

    # Load wav2vec 2.0 model..
    using_wav2vec2 = args.v2 and args.roberta is None and \
        args.vocab is not None
    if using_wav2vec2:
        modelf = str(args.modelf)
        model, cfg, task = \
            fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [modelf], arg_overrides={"data": args.vocab})
        wav2vec_model = model[0]
        wav2vec_model.eval()
        wav2vec_model = wav2vec_model.to(device)

    # Load wav2vec model..
    if not using_wav2vec2:
        wav2vec_cp = torch.load(args.modelf, map_location=torch.device('cpu'))
        with pipes() as (stderr, stdout):
            wav2vec_model = Wav2VecModel.build_model(
                wav2vec_cp['args'], task=None)

        wav2vec_model.load_state_dict(wav2vec_cp['model'])
        wav2vec_model.eval()
        wav2vec_model = wav2vec_model.to(device)

    # Process.
    with tqdm(total=len(args.afs)) as pbar:
        for fn in args.afs:
            npy_path = Path(args.feats_dir, fn.stem + '.npy')
            try:
                extract_feats_to_file(
                    npy_path, fn, wav2vec_model, roberta_model,
                    v2=using_wav2vec2)
            except RuntimeError:
                tqdm.write(
                    f'ERROR: CUDA OOM error when processing "{fn}". \
                      Skipping.')
                torch.cuda.empty_cache()

            pbar.update(1)


if __name__ == '__main__':
    main()
