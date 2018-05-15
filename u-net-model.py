
import torch
import torch.nn as nn
import torch.nn.functional as F



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(self).__init__()
        kern3 = 3
        kern2 = 2
        pad = samePad(kern3, 1)
        self.conv11 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = kern3, padding = pad)
        self.act12 = nn.ReLU()
        self.conv13 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = kern3, padding = pad)
        self.act14 = nn.ReLU()
        self.pool15 = nn.MaxPool2d(kern3el_size = 2)

        self.conv21 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = kern3, padding = pad)
        self.act22 = nn.ReLU()
        self.conv23 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = kern3, padding = pad)
        self.act24 = nn.ReLU()
        self.pool25 = nn.MaxPool2d(kern3el_size = 2)

        self.conv31 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = kern3, padding = pad)
        self.act32 = nn.ReLU()
        self.conv33 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = kern3, padding = pad)
        self.act34 = nn.ReLU()
        self.pool35 = nn.MaxPool2d(kern3el_size = 2)

        self.conv41 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = kern3, padding = pad)
        self.act42 = nn.ReLU()
        self.conv43 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = kern3, padding = pad)
        self.relu44 = nn.ReLU()
        self.pool45 = nn.MaxPool2d(kern3el_size = 2)

        self.conv51 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = kern3, padding = pad)
        self.act52 = nn.ReLU()
        self.conv53 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = kern3, padding = pad)
        self.act54 = nn.ReLU()

        pad2 = samePad(kern2, 2)
        self.convT61 = nn.ConvTranspose2d(in_channels = 128, out_channels =  64, kern_size = kern2, stride = 2, padding = pad2)
        self.cat62 = concatenate([self.convT61, self.act44])
        self.conv63 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = kern3, padding = pad)
        self.act64 = nn.ReLU()
        self.conv65 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = kern3, padding = pad)
        self.act66 = nn.ReLU()

        self.convT71 = nn.ConvTranspose2d(in_channels = 64, out_channels =  32, kern_size = kern2, stride = 2, padding = pad2)
        self.cat72 = concatenate([self.convT71, self.gact34])
        self.conv73 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = kern3, padding = pad)
        self.act74 = nn.ReLU()
        self.conv75 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = kern3, padding = pad)
        self.act76 = nn.ReLU()

        self.convT81 = nn.ConvTranspose2d(in_channels = 32, out_channels =  16, kern_size = kern2, stride = 2, padding = pad2)
        self.cat82 = concatenate([self.convT81, self.act24])
        self.conv83 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = kern3, padding = pad)
        self.act84 = nn.ReLU()
        self.conv85 = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = kern3, padding = pad)
        self.act86 = nn.ReLU()

        self.convT91 = nn.ConvTranspose2d(in_channels = 16, out_channels =  8, kern_size = kern2, stride = 2, padding = pad2)
        self.cat92 = concatenate([self.convT91, self.act14])
        self.conv93 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = kern3, padding = pad)
        self.act94 = nn.ReLU()
        self.conv95 = nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = kern3, padding = pad)
        self.act96 = nn.ReLU()


        self.output_conv = Conv2d(in_channels = 8, out_channels = 1, kernel_size = 1)
        self.output = nn.Sigmoid()

    def forward(self, input):
        x1 = self.conv11(input)
        x2 = self.act12(x1)
        x3 = self.conv13(x2)
        x4 = self.act14(x3)
        x5 = self.pool15(x4)

        x6 = self.conv21(x5)
        x7 = self.act22(x6)
        x8 = self.conv23(x7)
        x9 = self.act24(x8)
        x10 = self.pool25(x9)

        x11 = self.conv31(x10)
        x12 = self.act32(x11)
        x13 = self.conv33(x12)
        x14 = self.act34(x13)
        x15 = self.pool35(x14)

        x16 = self.conv41(x15)
        x17 = self.act42(x16)
        x18 = self.conv43(x17)
        x19 = self.act44(x18)
        x20 = self.pool45(x19)

        x21 = self.conv51(x20)
        x22 = self.act52(x21)
        x23 = self.conv53(x22)
        x24 = self.act54(x23)

        x25 = self.convT61(x24)
        x26 = self.cat62(x25)
        x27 = self.conv63(x26)
        x28 = self.act64(x27)
        x29 = self.conv65(x28)
        x30 = self.act66(x29)

        x31 = self.convT71(x30)
        x32 = self.cat72(x31)
        x34 = self.conv73(x32)
        x35 = self.act74(x33)
        x36 = self.conv75(x34)
        x37 = self.act76(x35)

        x38 = self.convT81(x37)
        x39 = self.cat82(x38)
        x40 = self.conv83(x39)
        x41 = self.act84(x40)
        x42 = self.conv85(x41)
        x43 = self.act86(x42)

        x44 = self.convT91(x43)
        x45 = self.cat92(x44)
        x46 = self.conv93(x45)
        x47 = self.act94(x46)
        x48 = self.conv95(x47)
        x49 = self.act96(x48)

        x50 = self.output_conv(x49)
        x51 = self.output(x50)
        return x51

    def samePad(filter, stride):
        return int(float(filter - stride)/2)
