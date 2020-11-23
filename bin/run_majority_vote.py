import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
from run_phone_classification import load_timit_phone_map


DATASET_LISTS = ['ctimit_final', 'ffmtimit_final', 'ntimit_final',
                 'stctimit_final', 'timit_final', 'wtimit_final']
PWD = Path('/scratch/dannima/pae_probe_experiments')


def main():
    parser = argparse.ArgumentParser(
        description='Measure the performance of majority vote.')
    parser.add_argument(
        '-t', dest='task', nargs=None, default='sad', type=str, metavar='STR',
        help='specify the task')
    args = parser.parse_args()
    records = []
    for dset in DATASET_LISTS:
        targets_path = PWD/'labels'/dset/args.task/'mel_fbank'/'test.npy'
        targets = np.load(targets_path)

        if args.task in ['sad', 'sonorant', 'syllable', 'fricative']:
            preds = np.ones(targets.shape, dtype=int)
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(
                targets, preds, pos_label=1, average='binary')

        elif args.task == 'phone':
            _, folded_index_map = load_timit_phone_map('phones.60-48-39.map')
            folded_targets = [folded_index_map[x] for x in targets]
            for phone in np.argsort(np.bincount(folded_targets))[::-1]:
                if phone != 30:
                    majority = phone
                    break
                else:
                    continue

            idx_nums = []
            for i, elem in enumerate(folded_targets):
                if elem != 30:
                    idx_nums.append(i)
            targets = [folded_targets[val] for val in idx_nums]
            preds = [majority] * len(targets)

            precision, recall, f1, _ = metrics.precision_recall_fscore_support(
                targets, preds, average='weighted')
        else:
            raise ValueError(f'Unrecognized task: "{args.task}".')

        acc = metrics.accuracy_score(targets, preds)
        records.append({
            'test': dset,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1})
    scores_df = pd.DataFrame(records)
    print(scores_df)


if __name__ == '__main__':
    main()
