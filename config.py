import argparse


class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args = None

    def default(self):
        self.parser.add_argument('--data_path', type=str, default='./data')
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--epoch', type=int, default=80)
        self.parser.add_argument('--learning_rate', type=float, default=0.001)

    def parse(self):
        self.default()
        self.args = self.parser.parse_args()

        return self.args
