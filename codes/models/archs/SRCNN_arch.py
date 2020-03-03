import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=9,padding=4);
        self.relu1 = nn.ReLU();
        self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=0);
        self.relu2 = nn.ReLU();
        self.conv3 = nn.Conv2d(32,3,kernel_size=5,padding=2);

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return out

if __name__ == '__main__':
    net = SRCNN()
    import torch
    x = torch.randn(2,1,32,32)
    y = net(x)
    print(y.size())


