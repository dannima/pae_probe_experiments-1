#!/usr/bin/env python3
import argparse
import os
import sys
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        'generate configuration files', add_help=True)
    parser.add_argument(
        'feats_dir', nargs=None,
        help='directory of extracted features')
    parser.add_argument(
        'config_dir', nargs=None,
        help='output directory for config files')
    parser.add_argument(
        'df', nargs='+',
        help='path to TIMIT variants')
    # parser.add_argument(
    #     '--ftype', nargs=None, default='mel_fbank', metavar='FEATURE',
    #     help='type of features (default: %(default)s)')
    # parser.add_argument(
    #     '--task', nargs=None, default='sad', metavar='TASK',
    #     help='probing tasks (default: %(default)s)')
    # parser.add_argument(
    #     '--clf', nargs=None, default='logistic', metavar='CLASSIFIER',
    #     help='classifier (default: %(default)s)')
    parser.add_argument(
        '--context_size', nargs=None, default=0, type=int,
        help='number of frames in each side as context to features \
              (default: %(default)s)')
    parser.add_argument(
        '--batch_size', nargs=None, default=1024, type=int,
        help='number of training examples used in one iteration \
              (default: %(default)s)')
    parser.add_argument(
        '--step', nargs=None, default=0.01, type=float, metavar='SECONDS',
        help='frame step in seconds (default: %(default)s)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    # print (args.df)
    # print (type(args.df))
    # print (args.df[0])
    # print (args.df[0]+'/train.ids')

    os.makedirs(args.config_dir, exist_ok=True)
    # args.config_dir.mkdir(parents=True, exist_ok=True)

    # Determine parameters for configuration files.
    for probing_task in ['sad', 'vowel', 'sonorant', 'fricative', 'phone']:
        for clf in ['logistic', 'max_margin', 'nnet']:
            data = dict(
                task=probing_task,
                classifier=clf,
                context_size=args.context_size,
                batch_size=args.batch_size,
                train_data=dict(
                    timit=dict(
                        uris=args.df[0]+'/train.ids',
                        step=args.step,
                        feats=args.feats_dir+'/timit',
                        phones=args.df[0]+'/phones'
                        ),
                    ntimit=dict(
                        uris=args.df[1]+'/train.ids',
                        step=args.step,
                        feats=args.feats_dir+'/ntimit',
                        phones=args.df[1]+'/phones'
                        ),
                    ctimit=dict(
                        uris=args.df[2]+'/train.ids',
                        step=args.step,
                        feats=args.feats_dir+'/ctimit',
                        phones=args.df[2]+'/phones'
                        ),
                    ffmtimit=dict(
                        uris=args.df[3]+'/train.ids',
                        step=args.step,
                        feats=args.feats_dir+'/ffmtimit',
                        phones=args.df[3]+'/phones'
                        ),
                    stctimit=dict(
                        uris=args.df[4]+'/train.ids',
                        step=args.step,
                        feats=args.feats_dir+'/stctimit',
                        phones=args.df[4]+'/phones'
                        ),
                    wtimit=dict(
                        uris=args.df[5]+'/train.ids',
                        step=args.step,
                        feats=args.feats_dir+'/wtimit',
                        phones=args.df[5]+'/phones'
                        )
                    ),
                test_data=dict(
                    timit=dict(
                        uris=args.df[0]+'/test_full.ids',
                        step=args.step,
                        feats=args.feats_dir+'/timit',
                        phones=args.df[0]+'/phones'
                        ),
                    ntimit=dict(
                        uris=args.df[1]+'/test_full.ids',
                        step=args.step,
                        feats=args.feats_dir+'/ntimit',
                        phones=args.df[1]+'/phones'
                        ),
                    ctimit=dict(
                        uris=args.df[2]+'/test_full.ids',
                        step=args.step,
                        feats=args.feats_dir+'/ctimit',
                        phones=args.df[2]+'/phones'
                        ),
                    ffmtimit=dict(
                        uris=args.df[3]+'/test_full.ids',
                        step=args.step,
                        feats=args.feats_dir+'/ffmtimit',
                        phones=args.df[3]+'/phones'
                        ),
                    stctimit=dict(
                        uris=args.df[4]+'/test_full.ids',
                        step=args.step,
                        feats=args.feats_dir+'/stctimit',
                        phones=args.df[4]+'/phones'
                        ),
                    wtimit=dict(
                        uris=args.df[5]+'/test_full.ids',
                        step=args.step,
                        feats=args.feats_dir+'/wtimit',
                        phones=args.df[5]+'/phones'
                        )
                    )
                )

            fn = Path('configs/tasks')/f'{probing_task}_{clf}.yaml'
            with open(fn, 'w') as config_file:
                yaml.dump(data, config_file, default_flow_style=False)


if __name__ == '__main__':
    main()
