import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv(x))
        out = self.activation(self.conv2(out))
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, padding=1)
        self.activation = activation

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.activation(self.conv(out))
        out = self.conv2(out)
        return out

class tiny_UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.activation = F.tanh
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block3_16 = UNetConvBlock(3, 16)
        self.conv_block16_32 = UNetConvBlock(16, 32)
        self.conv_block32_64 = UNetConvBlock(32, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)

        self.up_block128_64 = UNetUpBlock(128, 64)
        self.up_block64_32 = UNetUpBlock(64, 32)
        self.up_block32_16 = UNetUpBlock(32, 16)

        self.last = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        block1 = self.conv_block3_16(x)    # 16*64*64
        pool1 = self.pool1(block1)         # 16*32*32

        block2 = self.conv_block16_32(pool1)    # 32*32*32
        pool2 = self.pool2(block2)      # 32*16*16

        block3 = self.conv_block32_64(pool2)    # 64*16*16
        pool3 = self.pool3(block3)      #64*8*8

        block4 = self.conv_block64_128(pool3)       #128*8*8

        up1 = self.activation(self.up_block128_64(block4, block3))      # 16*16

        up2 = self.activation(self.up_block64_32(up1, block2))      #32*32

        up3 = self.up_block32_16(up2, block1)       #64*64

        return self.last(up3)       #64*64

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.activation = F.tanh
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.pool5 = nn.MaxPool2d(2)
        self.pool6 = nn.MaxPool2d(2)

        self.conv_block3_16 = UNetConvBlock(3, 16)
        self.conv_block16_32 = UNetConvBlock(16, 32)
        self.conv_block32_64 = UNetConvBlock(32, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        self.conv_block256_512 = UNetConvBlock(256, 512)

        self.conv_block512_1024 = UNetConvBlock(512, 1024)

        self.up_block1024_512 = UNetUpBlock(1024, 512)

        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)
        self.up_block64_32 = UNetUpBlock(64, 32)
        self.up_block32_16 = UNetUpBlock(32, 16)

        self.last = nn.Conv2d(16, 1, 1)
        
    def forward(self, x):
        block1 = self.conv_block3_16(x)    # 16*256*256
        pool1 = self.pool1(block1)         # 16*128*128

        block2 = self.conv_block16_32(pool1)    # 32*128*128
        pool2 = self.pool2(block2)      # 32*64*64

        block3 = self.conv_block32_64(pool2)    # 64*64*64
        pool3 = self.pool3(block3)      #64*32*32

        block4 = self.conv_block64_128(pool3)       #128*32*32
        
        pool4 = self.pool4(block4)# 128*16*16
        block5 = self.conv_block128_256(pool4)# 256*16*16
        pool5 = self.pool5(block5)# 256*8*8
        block6 = self.conv_block256_512(pool5)# 512*8*8
        pool6 = self.pool6(block6)# 512*4*4
        block7 = self.conv_block512_1024(pool6)# 1024*4*4

        upm2= self.activation(self.up_block1024_512(block7, block6))#512*8*8

        upm1= self.activation(self.up_block512_256(upm2, block5))#256*16*16
        up0 = self.activation(self.up_block256_128(upm1, block4))#128*32*32


        up1 = self.activation(self.up_block128_64(up0, block3))      #64*64*64

        up2 = self.activation(self.up_block64_32(up1, block2))      #32*128*128

        up3 = self.up_block32_16(up2, block1)       #16*256*256


        return self.last(up3)       #1*256*256        self.out = F.softmax()


class Sgmt_UNet(nn.Module):
    def __init__(self):
        super(Sgmt_UNet, self).__init__()

        self.activation = F.tanh
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.pool5 = nn.MaxPool2d(2)
        self.pool6 = nn.MaxPool2d(2)

        self.conv_block3_16 = UNetConvBlock(3, 16)
        self.conv_block16_32 = UNetConvBlock(16, 32)
        self.conv_block32_64 = UNetConvBlock(32, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        self.conv_block256_512 = UNetConvBlock(256, 512)

        self.conv_block512_1024 = UNetConvBlock(512, 1024)

        self.up_block1024_512 = UNetUpBlock(1024, 512)

        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)
        self.up_block64_32 = UNetUpBlock(64, 32)
        self.up_block32_16 = UNetUpBlock(32, 16)

        self.last = nn.Conv2d(16, 5, 1)
        
    def forward(self, x):
        block1 = self.conv_block3_16(x)    # 16*256*256
        pool1 = self.pool1(block1)         # 16*128*128

        block2 = self.conv_block16_32(pool1)    # 32*128*128
        pool2 = self.pool2(block2)      # 32*64*64

        block3 = self.conv_block32_64(pool2)    # 64*64*64
        pool3 = self.pool3(block3)      #64*32*32

        block4 = self.conv_block64_128(pool3)       #128*32*32
        
        pool4 = self.pool4(block4)# 128*16*16
        block5 = self.conv_block128_256(pool4)# 256*16*16
        pool5 = self.pool5(block5)# 256*8*8
        block6 = self.conv_block256_512(pool5)# 512*8*8
        pool6 = self.pool6(block6)# 512*4*4
        block7 = self.conv_block512_1024(pool6)# 1024*4*4

        upm2= self.activation(self.up_block1024_512(block7, block6))#512*8*8

        upm1= self.activation(self.up_block512_256(upm2, block5))#256*16*16
        up0 = self.activation(self.up_block256_128(upm1, block4))#128*32*32


        up1 = self.activation(self.up_block128_64(up0, block3))      #64*64*64

        up2 = self.activation(self.up_block64_32(up1, block2))      #32*128*128

        up3 = self.up_block32_16(up2, block1)       #16*256*256


        return self.last(up3)       #5*256*256        self.out = F.softmax()

class Plane_UNet(nn.Module):
    def __init__(self):
        super(Plane_UNet, self).__init__()

        self.activation = F.tanh
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.pool5 = nn.MaxPool2d(2)
        self.pool6 = nn.MaxPool2d(2)

        self.conv_block3_16 = UNetConvBlock(3, 16)
        self.conv_block16_32 = UNetConvBlock(16, 32)
        self.conv_block32_64 = UNetConvBlock(32, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        self.conv_block256_512 = UNetConvBlock(256, 512)

        self.conv_block512_1024 = UNetConvBlock(512, 1024)

        self.up_block1024_512 = UNetUpBlock(1024, 512)

        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)
        self.up_block64_32 = UNetUpBlock(64, 32)
        self.up_block32_16 = UNetUpBlock(32, 16)

        self.last = nn.Conv2d(16, 30, 1)
        
    def forward(self, x):
        block1 = self.conv_block3_16(x)    # 16*256*256
        pool1 = self.pool1(block1)         # 16*128*128

        block2 = self.conv_block16_32(pool1)    # 32*128*128
        pool2 = self.pool2(block2)      # 32*64*64

        block3 = self.conv_block32_64(pool2)    # 64*64*64
        pool3 = self.pool3(block3)      #64*32*32

        block4 = self.conv_block64_128(pool3)       #128*32*32
        
        pool4 = self.pool4(block4)# 128*16*16
        block5 = self.conv_block128_256(pool4)# 256*16*16
        pool5 = self.pool5(block5)# 256*8*8
        block6 = self.conv_block256_512(pool5)# 512*8*8
        pool6 = self.pool6(block6)# 512*4*4
        block7 = self.conv_block512_1024(pool6)# 1024*4*4

        upm2= self.activation(self.up_block1024_512(block7, block6))#512*8*8

        upm1= self.activation(self.up_block512_256(upm2, block5))#256*16*16
        up0 = self.activation(self.up_block256_128(upm1, block4))#128*32*32


        up1 = self.activation(self.up_block128_64(up0, block3))      #64*64*64

        up2 = self.activation(self.up_block64_32(up1, block2))      #32*128*128

        up3 = self.up_block32_16(up2, block1)       #16*256*256


        return self.last(up3)       #16*256*256        self.out = F.softmax()

class Normal_UNet(nn.Module):
    def __init__(self):
        super(Normal_UNet, self).__init__()

        self.activation = F.tanh
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.pool5 = nn.MaxPool2d(2)
        self.pool6 = nn.MaxPool2d(2)

        self.conv_block3_16 = UNetConvBlock(3, 16)
        self.conv_block16_32 = UNetConvBlock(16, 32)
        self.conv_block32_64 = UNetConvBlock(32, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        self.conv_block256_512 = UNetConvBlock(256, 512)

        self.conv_block512_1024 = UNetConvBlock(512, 1024)

        self.up_block1024_512 = UNetUpBlock(1024, 512)

        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)
        self.up_block64_32 = UNetUpBlock(64, 32)
        self.up_block32_16 = UNetUpBlock(32, 16)

        self.last = nn.Conv2d(16, 3, 1)
        
    def forward(self, x):
        block1 = self.conv_block3_16(x)    # 16*256*256
        pool1 = self.pool1(block1)         # 16*128*128

        block2 = self.conv_block16_32(pool1)    # 32*128*128
        pool2 = self.pool2(block2)      # 32*64*64

        block3 = self.conv_block32_64(pool2)    # 64*64*64
        pool3 = self.pool3(block3)      #64*32*32

        block4 = self.conv_block64_128(pool3)       #128*32*32
        
        pool4 = self.pool4(block4)# 128*16*16
        block5 = self.conv_block128_256(pool4)# 256*16*16
        pool5 = self.pool5(block5)# 256*8*8
        block6 = self.conv_block256_512(pool5)# 512*8*8
        pool6 = self.pool6(block6)# 512*4*4
        block7 = self.conv_block512_1024(pool6)# 1024*4*4

        upm2= self.activation(self.up_block1024_512(block7, block6))#512*8*8

        upm1= self.activation(self.up_block512_256(upm2, block5))#256*16*16
        up0 = self.activation(self.up_block256_128(upm1, block4))#128*32*32


        up1 = self.activation(self.up_block128_64(up0, block3))      #64*64*64

        up2 = self.activation(self.up_block64_32(up1, block2))      #32*128*128

        up3 = self.up_block32_16(up2, block1)       #16*256*256

        lastt = self.last(up3) 
        self.last2 =  F.normalize(lastt, p=2, dim=0)
        print(self.last2)
        return  self.last2     #16*256*256        self.out = F.softmax()