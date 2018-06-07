import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
#from .unet_parts import *

class UNet_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        pad = samePad(3, 1)
        self.conv11 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv11.weight)
        self.conv13 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv13.weight)
        self.pool15 = nn.MaxPool2d(kernel_size = 2)

        # Imagnet kicks in
        imgNetModel = torch.load('imgNetModel')

        self.conv21 = imgNetModel.conv21
        self.conv23 = imgNetModel.conv23
        self.pool25 = imgNetModel.pool25

        self.conv31 = imgNetModel.conv31
        self.conv33 = imgNetModel.conv33
        self.pool35 = imgNetModel.pool35

        self.conv41 = imgNetModel.conv41
        self.conv43 = imgNetModel.conv43
        self.pool45 = imgNetModel.pool45
        
        self.conv51 = imgNetModel.conv51
        self.conv53 = imgNetModel.conv53
        
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
        self.convT91 = nn.ConvTranspose2d(in_channels = 16, out_channels =  8, kernel_size = 2, stride = 2, output_padding = 0)
        nn.init.kaiming_normal_(self.convT91.weight)
        
        self.convT91 = nn.ConvTranspose2d(in_channels = 16, out_channels =  8, kernel_size = 2, stride = 2, output_padding = 0)
        nn.init.kaiming_normal_(self.convT91.weight)
        
        self.conv93 = nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv93.weight)
        self.conv95 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv95.weight)
        self.conv96 = nn.Conv2d(in_channels = 8, out_channels = 1, kernel_size = 1, padding = 0)
        nn.init.kaiming_normal_(self.conv96.weight)
        self.output_fn = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv13(F.relu(x1))
        x3 = self.pool15(F.relu(x2))

        x4 = self.conv21(x3)
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

        x24 = self.convT91(F.relu(x23))
        x25 = self.conv93(torch.cat((x2, x24), 1))
        x26 = self.conv95(F.relu(x25))

        x27 = self.conv96(F.relu(x26))
        x_out = self.output_fn(x27)
        return x_out


def samePad(filterSize, stride):
    return int(float(filterSize - stride)/2)
