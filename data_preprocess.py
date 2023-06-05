import os.path
import threading

import numpy as np
import pandas as pd
import librosa


def data_cleaning():
    annotations = pd.read_table('./data/annotations_final.csv',
                                usecols=(0, 28, 34, 42, 49, 69, 72, 83, 91, 94, 100, 114,
                                         124, 140, 146, 154, 159, 168, 180, 181, 182, 189))

    for i in range(0, len(annotations)):
        row = annotations.loc[i]
        if sum(row[1:-1]) == 0:
            print(f"clip {row['clip_id']} is invalid, dropping it.")
            annotations.drop(i, axis=0, inplace=True)

    print(f'data cleaning finished, saving result.')
    annotations.to_csv('./data/annotations_cleaned.csv')
    print(f'cleaned data saved.')


def generate_dataset():
    dataset = np.zeros(shape=(11534, 1, 24, 1255), dtype=np.float32)
    annotations = np.array(pd.read_csv('./data/annotations_cleaned.csv'))[:, 1:]
    audio_paths = annotations[:, -1]
    # dataset = np.zeros(shape=(10, 1, 24, 1255), dtype=np.float32)
    # annotations = np.array(pd.read_csv('./data/annotations_cleaned.csv'))[:10, 1:]
    # audio_paths = annotations[:10, -1]
    for index, path in enumerate(audio_paths):
        try:
            y, sr = librosa.load(os.path.join('./data', 'mp3', path))
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24)
            temp = np.zeros(shape=(1, 24, 1255), dtype=np.float32)
            temp[0] = mfccs
            dataset[index] = temp
        except Exception:
            print(f'Load file {path} failed, index: {index}')
            continue

    print(f'Dataset generated.')
    return dataset
