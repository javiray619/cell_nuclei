import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
#from .unet_parts import *

class GroupNormUNet(nn.Module):
    def __init__(self):
        super().__init__()
        pad = samePad(3, 1)
        self.conv11 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = pad)        
        nn.init.kaiming_normal_(self.conv11.weight)
        # Separate 8 channels into 4 groups
        self.group11 = nn.GroupNorm(2, 8)
        self.conv13 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv13.weight)
        self.group13 = nn.GroupNorm(2, 8)
        self.pool15 = nn.MaxPool2d(kernel_size = 2)

        self.conv21 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv21.weight)
        self.group21 = nn.GroupNorm(4, 16)
        self.conv23 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv23.weight)
        self.group23 = nn.GroupNorm(4, 16)        
        self.pool25 = nn.MaxPool2d(kernel_size = 2)

        self.conv31 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv31.weight)
        self.group31 = nn.GroupNorm(8, 32)
        self.conv33 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv33.weight)
        self.group33 = nn.GroupNorm(8, 32)
        self.pool35 = nn.MaxPool2d(kernel_size = 2)

        self.conv41 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv41.weight)
        self.group41 = nn.GroupNorm(16, 64)        
        self.conv43 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv43.weight)
        self.group43 = nn.GroupNorm(16, 64)
        self.pool45 = nn.MaxPool2d(kernel_size = 2)

        self.conv51 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv51.weight)
        self.group51 = nn.GroupNorm(32, 128)
        self.conv53 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv53.weight)
        self.group53 = nn.GroupNorm(32, 128)
        self.convT61 = nn.ConvTranspose2d(in_channels = 128, out_channels =  64, kernel_size = 2, stride = 2, output_padding = 0)
        nn.init.kaiming_normal_(self.convT61.weight)
        self.group61 = nn.GroupNorm(16, 64)
        
        self.conv63 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv63.weight)
        self.group63 = nn.GroupNorm(16, 64)
        self.conv65 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv65.weight)
        self.group65 = nn.GroupNorm(16, 64)
        self.convT71 = nn.ConvTranspose2d(in_channels = 64, out_channels =  32, kernel_size = 2, stride = 2, output_padding = 0)
        nn.init.kaiming_normal_(self.convT71.weight)
        self.group71 = nn.GroupNorm(8, 32)
       
        self.conv73 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv73.weight)
        self.group73 = nn.GroupNorm(8, 32)
        self.conv75 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv75.weight)
        self.group75 = nn.GroupNorm(8, 32)
        self.convT81 = nn.ConvTranspose2d(in_channels = 32, out_channels =  16, kernel_size = 2, stride = 2, output_padding = 0)
        nn.init.kaiming_normal_(self.convT81.weight)
        self.group81 = nn.GroupNorm(4, 16)
       
        self.conv83 = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv83.weight)
        self.group83 = nn.GroupNorm(4, 16)
        self.conv85 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv85.weight)
        self.group85 = nn.GroupNorm(4, 16)
        self.convT91 = nn.ConvTranspose2d(in_channels = 16, out_channels =  8, kernel_size = 2, stride = 2, output_padding = 0)
        nn.init.kaiming_normal_(self.convT91.weight)
        self.group91 = nn.GroupNorm(2, 8)

        
        self.conv93 = nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv93.weight)
        self.group93 = nn.GroupNorm(2, 8)
        self.conv95 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, padding = pad)
        nn.init.kaiming_normal_(self.conv95.weight)
        self.group95 = nn.GroupNorm(2, 8)
        self.conv96 = nn.Conv2d(in_channels = 8, out_channels = 1, kernel_size = 1, padding = 0)
        nn.init.kaiming_normal_(self.conv96.weight)
        self.layer96 = nn.LayerNorm([1, 128, 128])        
        self.output_fn = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv11(x)
        x2 = self.conv13(F.relu(self.group11(x1)))
        x3 = self.pool15(F.relu(self.group13(x2)))

        x4 = self.conv21(x3)
        x5 = self.conv23(F.relu(self.group21(x4)))
        x6 = self.pool25(F.relu(self.group23(x5)))

        x7 = self.conv31(x6)
        x8 = self.conv33(F.relu(self.group31(x7)))
        x9 = self.pool35(F.relu(self.group33(x8)))

        x10 = self.conv41(x9)
        x11 = self.conv43(F.relu(self.group41(x10)))
        x12 = self.pool45(F.relu(self.group43(x11)))

        x13 = self.conv51(x12)
        x14 = self.conv53(F.relu(self.group51(x13)))

        x15 = self.convT61(F.relu(self.group53(x14)))
        x16_input = torch.cat((x11, self.group61(x15)), 1)
        x16 = self.conv63(x16_input)
        x17 = self.conv65(F.relu(self.group63(x16)))

        x18 = self.convT71(F.relu(self.group65(x17)))
        x19 = self.conv73(torch.cat((x8, self.group71(x18)), 1))
        x20 = self.conv75(F.relu(self.group73(x19)))

        x21 = self.convT81(F.relu(self.group75(x20)))
        x22 = self.conv83(torch.cat((x5, self.group81(x21)), 1))
        x23 = self.conv85(F.relu(self.group83(x22)))

        x24 = self.convT91(F.relu(self.group85(x23)))
        x25 = self.conv93(torch.cat((x2, self.group91(x24)), 1))
        x26 = self.conv95(F.relu(self.group93(x25)))

        x27 = self.conv96(F.relu(self.group95(x26)))
        x_out = self.output_fn(self.layer96(x27))
        return x_out

def samePad(filterSize, stride):
    return int(float(filterSize - stride)/2)
