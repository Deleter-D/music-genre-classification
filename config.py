import argparse


class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args = None

    def default(self):
        self.parser.add_argument('--data_path', type=str, default='./data')

    def parse(self):
        self.default()
        self.args = self.parser.parse_args()

        return self.args
