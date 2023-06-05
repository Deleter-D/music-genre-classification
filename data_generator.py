import os.path
from threading import Thread, Semaphore

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader

from config import Config

args = Config().parse()


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
    y, sr = librosa.load(os.path.join(args.data_path, 'mp3', audio_path))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=24)
    res = np.zeros(shape=(1, 24, 1255), dtype=np.float32)
    res[0] = mfccs
    return res


def generate_dataloader():
    data = np.zeros(shape=(11534, 1, 24, 1255), dtype=np.float32)
    annotations = np.array(pd.read_csv(os.path.join(args.data_path, 'annotations_cleaned.csv')))[:, 1:]
    audio_paths = annotations[:, -1]

    # Max thread count
    semaphore = Semaphore(128)
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

    annotations = pd.read_csv(os.path.join(args.data_path, 'annotations_cleaned.csv'),
                              usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21))
    annotations = np.array(annotations, dtype=np.float32)
    x = torch.from_numpy(data)
    y = torch.from_numpy(annotations)
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    return data_loader
