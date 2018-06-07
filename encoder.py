
import torch
import torch.nn as nn
import torch.nn.functional as F

# Model based on CellNuclei.ipynb reference [1]

class encoderNet(nn.Module):
    def __init__(self):
        super().__init__()
        pad = samePad(3, 1)
        self.conv11 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv11.weight)


        ### WEIGHTS BELOW USED FOR TRANSFER LEARNING ###
        self.conv21 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv21.weight)
        self.conv23 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv23.weight)
        self.pool25 = nn.MaxPool2d(kernel_size = 2)

        self.conv31 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv31.weight)
        self.conv33 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv33.weight)
        self.pool35 = nn.MaxPool2d(kernel_size = 2)

        self.conv41 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv41.weight)
        self.conv43 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv43.weight)
        self.pool45 = nn.MaxPool2d(kernel_size = 2)

        self.conv51 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv51.weight)
        self.conv53 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv53.weight)

        ### WEIGHTS ABOVE USED FOR TRANSFER LEARNING ###


        self.convT61 = nn.ConvTranspose2d(in_channels = 128, out_channels =  64, kernel_size = 2, stride = 2, output_padding = 0)
        nn.init.kaiming_normal_(self.convT61.weight)
        
        self.conv63 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv63.weight)
        self.conv65 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv65.weight)
        self.convT71 = nn.ConvTranspose2d(in_channels = 64, out_channels =  32, kernel_size = 2, stride = 2, output_padding = 0)
        nn.init.kaiming_normal_(self.convT71.weight)
        
        self.conv73 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv73.weight)
        self.conv75 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv75.weight)
        self.convT81 = nn.ConvTranspose2d(in_channels = 32, out_channels =  16, kernel_size = 2, stride = 2, output_padding = 0)
        nn.init.kaiming_normal_(self.convT81.weight)
        
        self.conv83 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv83.weight)
        self.conv85 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv85.weight)

        self.convFC1 = nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 1, padding = 0)
        nn.init.kaiming_normal_(self.convFC1.weight)
        self.outputLinear = nn.Linear(4096, 1000)
        nn.init.kaiming_normal_(self.outputLinear.weight)

    def forward(self, x):
        x1 = self.conv11(x)
        x4 = self.conv21(F.relu(x1))
        
        x5 = self.conv23(F.relu(x4))
        x6 = self.pool25(F.relu(x5))

        x7 = self.conv31(x6)
        x8 = self.conv33(F.relu(x7))
        x9 = self.pool35(F.relu(x8))

        x10 = self.conv41(x9)
        x11 = self.conv43(F.relu(x10))
        x12 = self.pool45(F.relu(x11))

        x13 = self.conv51(x12)
        x14 = self.conv53(F.relu(x13))

        x15 = self.convT61(F.relu(x14))
        x16_input = torch.cat((x11, x15), 1)
        x16 = self.conv63(x16_input)
        x17 = self.conv65(F.relu(x16))

        x18 = self.convT71(F.relu(x17))
        x19 = self.conv73(torch.cat((x8, x18), 1))
        x20 = self.conv75(F.relu(x19))

        x21 = self.convT81(F.relu(x20))
        x22 = self.conv83(torch.cat((x5, x21), 1))
        x23 = self.conv85(F.relu(x22))


        x24 = self.convFC1(F.relu(x23))
        x25 = flatten(F.relu(x24.squeeze()))

        return self.outputLinear(x25)

def samePad(filterSize, stride):
    return int(float(filterSize - stride)/2)

def flatten(x):
    N, W, H = x.shape
    return x.view(N, W*H)
