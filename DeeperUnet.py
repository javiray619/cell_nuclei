import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
#from .unet_parts import *

class DeeperUNet(nn.Module):
    def __init__(self):
        super().__init__()
        pad = samePad(3, 1)
        self.conv11 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv11.weight)
        self.conv12 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv12.weight)
        self.conv13 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv13.weight)
        self.conv14 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv14.weight)
        self.pool15 = nn.MaxPool2d(kernel_size = 2)

        self.conv21 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv21.weight)
        self.conv22 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv22.weight)
        self.conv23 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv23.weight)
        self.conv24 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv24.weight)
        self.pool25 = nn.MaxPool2d(kernel_size = 2)

        self.conv31 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv31.weight)
        self.conv32 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv32.weight)
        self.conv33 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv33.weight)
        self.conv34 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv34.weight)
        self.pool35 = nn.MaxPool2d(kernel_size = 2)

        self.conv41 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv41.weight)
        self.conv42 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv42.weight)
        self.conv43 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv43.weight)
        self.conv44 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv44.weight)
        self.pool45 = nn.MaxPool2d(kernel_size = 2)

        self.conv51 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv51.weight)
        self.conv52 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv52.weight)
        self.conv53 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv53.weight)
        self.conv54 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv54.weight)
        self.convT61 = nn.ConvTranspose2d(in_channels = 128, out_channels =  64, kernel_size = 2, stride = 2, output_padding = 0)
        nn.init.kaiming_normal_(self.convT61.weight)
        
        self.conv63 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv63.weight)
        self.conv64 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv64.weight)
        self.conv65 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv65.weight)
        self.conv66 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv66.weight)
        self.convT71 = nn.ConvTranspose2d(in_channels = 64, out_channels =  32, kernel_size = 2, stride = 2, output_padding = 0)
        nn.init.kaiming_normal_(self.convT71.weight)
        
        self.conv73 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv73.weight)
        self.conv74 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv74.weight)
        self.conv75 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv75.weight)
        self.conv76 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv76.weight)
        self.convT81 = nn.ConvTranspose2d(in_channels = 32, out_channels =  16, kernel_size = 2, stride = 2, output_padding = 0)
        nn.init.kaiming_normal_(self.convT81.weight)
        
        self.conv83 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv83.weight)
        self.conv84 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv84.weight)
        self.conv85 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv85.weight)
        self.conv86 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv86.weight)
        self.convT91 = nn.ConvTranspose2d(in_channels = 16, out_channels =  8, kernel_size = 2, stride = 2, output_padding = 0)
        nn.init.kaiming_normal_(self.convT91.weight)
        
        self.conv93 = nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv93.weight)
        self.conv94 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv94.weight)
        self.conv95 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv95.weight)
        self.conv96 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv96.weight)
        self.conv97 = nn.Conv2d(in_channels = 8, out_channels = 1, kernel_size = 1, padding = 0)
        nn.init.kaiming_normal_(self.conv97.weight)
        self.output_fn = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv11(x)
        x1d = self.conv12(F.relu(x1))
        x2 = self.conv13(F.relu(x1d))
        x2d = self.conv14(F.relu(x2))
        x3 = self.pool15(F.relu(x2d))

        x4 = self.conv21(x3)
        x4d = self.conv22(F.relu(x4))                  
        x5 = self.conv23(F.relu(x4d))
        x5d = self.conv24(F.relu(x5))
        x6 = self.pool25(F.relu(x5d))

        x7 = self.conv31(x6)
        x7d = self.conv32(F.relu(x7))
        x8 = self.conv33(F.relu(x7d))
        x8d = self.conv34(F.relu(x8))
        x9 = self.pool35(F.relu(x8d))

        x10 = self.conv41(x9)
        x10d = self.conv42(F.relu(x10))
        x11 = self.conv43(F.relu(x10d))
        x11d = self.conv44(F.relu(x11))
        x12 = self.pool45(F.relu(x11d))

        x13 = self.conv51(x12)
        x13d = self.conv52(F.relu(x13))
        x14 = self.conv53(F.relu(x13d))
        x14d = self.conv54(F.relu(x14))
                          
        x15 = self.convT61(F.relu(x14d))
        x16_input = torch.cat((x11d, x15), 1)
        x16 = self.conv63(x16_input)
        x16d = self.conv64(F.relu(x16)) 
        x17 = self.conv65(F.relu(x16d))
        x17d = self.conv66(F.relu(x17))

        x18 = self.convT71(F.relu(x17d))
        x19 = self.conv73(torch.cat((x8d, x18), 1))
        x19d = self.conv74(F.relu(x19))
        x20 = self.conv75(F.relu(x19d))
        x20d = self.conv76(F.relu(x20))

        x21 = self.convT81(F.relu(x20d))
        x22 = self.conv83(torch.cat((x5, x21), 1))
        x22d = self.conv84(F.relu(x22))
        x23 = self.conv85(F.relu(x22d))
        x23d = self.conv86(F.relu(x23))

        x24 = self.convT91(F.relu(x23d))
        x25 = self.conv93(torch.cat((x2d, x24), 1))
        x25d = self.conv94(F.relu(x25))
        x26 = self.conv95(F.relu(x25d))
        x26d = self.conv96(F.relu(x26))                  

        x27 = self.conv97(F.relu(x26d))
        x_out = self.output_fn(x27)
        return x_out

def samePad(filterSize, stride):
    return int(float(filterSize - stride)/2)