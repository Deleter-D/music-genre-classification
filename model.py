from torch.nn import Module, Conv2d, MaxPool2d, Linear, Flatten
from torch.nn.functional import relu


class Net(Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool4 = MaxPool2d(kernel_size=2, stride=2)
        self.fc5 = Linear(16 * 3 * 310, 1024)
        self.fc6 = Linear(1024, 512)
        self.fc7 = Linear(512, 256)
        self.fc8 = Linear(256, 64)
        self.fc9 = Linear(64, 20)
        self.flatten = Flatten()

    def forward(self, x):
        x = relu(self.conv1(x))
        x = self.pool2(x)
        x = relu(self.conv3(x))
        x = self.pool4(x)
        x = self.flatten(x)
        x = relu(self.fc5(x))
        x = relu(self.fc6(x))
        x = relu(self.fc7(x))
        x = relu(self.fc8(x))
        x = self.fc9(x)
        return x
