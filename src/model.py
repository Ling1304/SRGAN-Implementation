import torch
from torch import nn

# SOURCE: 
# - 
# - https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

class Gen_Residual_Block(nn.Module): 
    """
    Residual block for Generator Network
    """
    def __init__(self, channels):
        super().__init__() 
        # First convolutional layer
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1)
        # First batch normalization
        self.batch_norm1 = nn.BatchNorm2d(channels)
        # Parametic ReLU (PReLU)
        self.prelu = nn.PReLU()
        # Second convolutional layer
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1)
        # Second batch normalization
        self.batch_norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        output = self.conv1(x)
        output = self.batch_norm1(x)
        output = self.prelu(x)
        output = self.conv2(x)
        output = self.batch_norm2(x)

        # Return element wise sum of input x and output
        return output + x

class Sub_Pixel(nn.Module): 
    """
    Trained sub-pixel convolution layers
    """
    def __init__(self, channels, upscale_factor):
        super().__init__()
        # Convolutional layer
        self.conv1 = nn.Conv2d(channels, channels * upscale_factor **2, kernel_size=3, stride=1)
        # Shuffle the pixels based on uspscale factor (e.g. x2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        # PReLU 
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)

        return x

class Discrim_Block(nn.Module): 
    """
    Convolutional block for Discriminator
    """
    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        # Convolutional layer
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride)
        # Batch normalization
        self.batch_norm = nn.BatchNorm2d(out_channel)
        # Leaky ReLU 
        self.leaky_ReLU = nn.LeakyReLU()

    def forward(self, x): 
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.leaky_ReLU(x)

        return x

class Generator(nn.Module): 
    """
    Generator Network 
    """
    def __init__(self):
        super().__init__()
        # Initial convolutional layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1),
            nn.PReLU()
        )

        # 16 Residual Blocks with skip connections
        self.block1 = Gen_Residual_Block(64)
        self.block2 = Gen_Residual_Block(64)
        self.block3 = Gen_Residual_Block(64)
        self.block4 = Gen_Residual_Block(64)
        self.block5 = Gen_Residual_Block(64)
        self.block6 = Gen_Residual_Block(64)
        self.block7 = Gen_Residual_Block(64)
        self.block8 = Gen_Residual_Block(64)

        self.block9 = Gen_Residual_Block(64)
        self.block10 = Gen_Residual_Block(64)
        self.block11 = Gen_Residual_Block(64)
        self.block12 = Gen_Residual_Block(64)
        self.block13 = Gen_Residual_Block(64)
        self.block14 = Gen_Residual_Block(64)
        self.block15 = Gen_Residual_Block(64)
        self.block16 = Gen_Residual_Block(64)

        # Last block 
        # - No PreLU
        # - No second 'Conv2d' and 'BatchNorm2d')
        # - Element wise sum with output of initial convolutional layer (line 146)
        self.last_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64)
        )
        
        # 2 trained sub-pixel convolution layers (means 4x upscale)
        self.subpix_layer1 = Sub_Pixel(64, 4)
        self.subpix_layer2 = Sub_Pixel(64, 4)

        # Final convolutional layer
        self.conv_fin = nn.Conv2d(64, 3, kernel_size=9, stride=1)

    def foward(self, x): 
        # Initial convolutional layer
        conv1 = self.initial_conv(x)

        # 16 residual blocks
        b1 = self.block1(conv1)
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        b5 = self.block5(b4)
        b6 = self.block6(b5)
        b7 = self.block7(b6)
        b8 = self.block8(b7)

        b9 = self.block9(b8)
        b10 = self.block10(b9)
        b11 = self.block11(b10)
        b12 = self.block12(b11)
        b13 = self.block13(b12)
        b14 = self.block14(b13)
        b15 = self.block15(b14)
        b16 = self.block16(b15)

        # Last block
        last_b = self.last_block(b16) + conv1 # Element wise sum with output of initial convolutional layer
        
        # 2 trained sub-pixel convolution layers (means 4x upscale)
        subpix1 = self.subpix_layer1(last_b) 

        # Final convolutional layer
        fin_output = self.conv_fin(subpix1)

        # Normalize to range [0-1]
        return (torch.tanh(fin_output) + 1) / 2

class Discriminator(nn.Module): 
    """
    Discriminator Network
    """
    def __init__(self):
        super().__init__()

        # Initial Layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.LeakyReLU()
        )

        # 7 Discriminator blocks
        self.block1 = Discrim_Block(64, 64, stride=2)
        self.block2 = Discrim_Block(64, 128, stride=1)
        self.block3 = Discrim_Block(128, 128, stride=2)
        self.block4 = Discrim_Block(128, 256, stride=1)
        self.block5 = Discrim_Block(256, 256, stride=2)
        self.block6 = Discrim_Block(256, 512, stride=1)
        self.block7 = Discrim_Block(512, 512, stride=2)

        # First dense layer
        self.dense1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU()
        )

        # Second dense layer
        self.dense2 = nn.Sequential(
            nn.Linear(1024, 1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial_conv(x)
        
        b1 = self.block1(x) 
        b2 = self.block2(b1)
        b3 = self.block3(b2)
        b4 = self.block4(b3)
        b5 = self.block5(b4)
        b6 = self.block6(b5)
        b7 = self.block7(b6)

        dense1 = self.dense1(b7)
        fin_output = self.dense2(dense1)

        return fin_output

