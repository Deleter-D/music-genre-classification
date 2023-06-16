import os.path
from threading import Thread, Semaphore

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader

from config import Config

cmd_args = Config().parse()


class MFCCThread(Thread):
    def __init__(self, func, args):
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def getResult(self):
        return self.result


def get_mfcc(audio_path):
    y, sr = librosa.load(os.path.join(cmd_args.data_path, 'mp3', audio_path))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32, n_fft=800, hop_length=200)[:, :3200]
    res = np.zeros(shape=(1, 32, 3200), dtype=np.float32)
    res[0] = mfccs
    return res


def generate_dataloader():
    data = np.zeros(shape=(11534, 1, 32, 3200), dtype=np.float32)
    annotations = np.array(pd.read_csv(os.path.join(cmd_args.data_path, 'annotations_cleaned.csv')))[:, 1:]
    audio_paths = annotations[:, -1]

    # Max thread count
    semaphore = Semaphore(256)
    with tqdm(total=11534) as pbar:
        pbar.set_description('Generating MFCCs')
        for index, path in enumerate(audio_paths):
            with semaphore:
                try:
                    t = MFCCThread(func=get_mfcc, args=(path,))
                    t.start()
                    t.join()
                    data[index] = t.result
                    pbar.update()
                except Exception:
                    print(f'Load file {path} failed, index: {index}')
                    continue

    annotations = pd.read_csv(os.path.join(cmd_args.data_path, 'annotations_cleaned.csv'),
                              usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21))
    annotations = np.array(annotations, dtype=np.float32)
    x = torch.from_numpy(data)
    y = torch.from_numpy(annotations)
    dataset = TensorDataset(x, y)
    train_db, val_db = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(dataset=train_db, batch_size=cmd_args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_db, batch_size=cmd_args.batch_size, shuffle=True)

    return train_loader, val_loader
