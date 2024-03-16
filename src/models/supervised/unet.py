import torch
import torch.nn as nn
from torch.nn.functional import relu, pad

class DoubleConvHelper(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Module that implements 
            - a convolution
            - a batch norm
            - relu
            - another convolution
            - another batch norm
        
        Input:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            mid_channels (int): number of channels to use in the intermediate layer    
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass through the layers of the helper block"""
        return self.double_conv(x)

class Encoder(nn.Module):
    """ Downscale using the maxpool then call double conv helper. """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvHelper(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Decoder(nn.Module):
    """ Upscale using ConvTranspose2d then call double conv helper. """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upscale = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=3, stride=2)
        self.conv = DoubleConvHelper(in_channels, out_channels)

    def forward(self, x1, x2):
        """ 
        1) x1 is passed through either the upsample or the convtranspose2d
        2) The difference between x1 and x2 is calculated to account for differences in padding
        3) x1 is padded (or not padded) accordingly
        4) x2 represents the skip connection
        5) Concatenate x1 and x2 together with torch.cat
        6) Pass the concatenated tensor through a doubleconvhelper
        7) Return output
        """
        x1 = self.upscale(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_encoders: int = 2,
                 embedding_size: int = 64, scale_factor: int = 50, **kwargs):
        """
        Implements a unet, a network where the input is downscaled
        down to a lower resolution with a higher amount of channels,
        but the residual images between encoders are saved
        to be concatednated to later stages, creatin the
        nominal "U" shape.

        In order to do this, we will need n_encoders-1 encoders. 
        The first layer will be a doubleconvhelper that
        projects the in_channels image to an embedding_size
        image of the same size.

        After that, n_encoders-1 encoders are used which halve
        the size of the image, but double the amount of channels
        available to them (i.e, the first layer is 
        embedding_size -> 2*embedding size, the second layer is
        2*embedding_size -> 4*embedding_size, etc)

        The decoders then upscale the image and halve the amount of
        embedding layers, i.e., they go from 4*embedding_size->2*embedding_size.

        We then have a maxpool2d that scales down the output to by scale_factor,
        as the input for this architecture must be the same size as the output,
        but our input images are 800x800 and our output images are 16x16.

        Input:
            in_channels: number of input channels of the image
            of shape (batch, in_channels, width, height)
            out_channels: number of output channels of prediction,
            prediction is shape (batch, out_channels, width//scale_factor, height//scale_factor)
            n_encoders: number of encoders to use in the network (implementing this parameter is
            optional, but it is a good idea to have it as a parameter in case you want to experiment,
            if you do not implement it, you can assume n_encoders=2)
            embedding_size: number of channels to use in the first layer
            scale_factor: number of input pixels that map to 1 output pixel,
            for example, if the input is 800x800 and the output is 16x6
            then the scale factor is 800/16 = 50.
        """
        super(UNet, self).__init__()
        self.n_encoders = n_encoders
        self.inc = DoubleConvHelper(in_channels, embedding_size)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.outc = nn.Conv2d(embedding_size, out_channels, kernel_size=1)
        self.final_downscale = nn.MaxPool2d(scale_factor)

        for i in range(n_encoders):
            self.downs.append(Encoder(embedding_size * 2 ** i, embedding_size * 2 ** (i + 1)))

        for i in range(n_encoders, 0, -1):
            self.ups.append(Decoder(embedding_size * 2 ** i, embedding_size * 2 ** (i - 1)))

    def forward(self, x):
        """
            The image is passed through the encoder layers,
            making sure to save the residuals in a list.

            Following this, the residuals are passed to the
            decoder in reverse, excluding the last residual
            (as this is used as the input to the first decoder).

            The ith decoder should have an input of shape
            (batch, some_embedding_size, some_width, some_height)
            as the input image and
            (batch, some_embedding_size//2, 2*some_width, 2*some_height)
            as the residual.
        """
        x1 = self.inc(x)
        xs = [x1]
        for down in self.downs:
            xs.append(down(xs[-1]))
        x = xs.pop()
        for up in self.ups:
            x = up(x, xs.pop())
        x = self.outc(x)
        logits = self.final_downscale(x)
        return logits
