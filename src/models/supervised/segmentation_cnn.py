import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List

class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, depth: int, kernel_size: int, pool_size: int):
        """
        This class represents a CNN block which projects
        `in_channels' to a usually higher amount of channels
        `out_channels'. It runs the image through `depth' layers
        with a kernel of size `kernel_size'. In order to keep
        the resolution of the image the same, it must be padded
        by with `kernel_size//2' zeroes around the image. 

        The image is then pooled with a MaxPool2d operation of size
        `pool_size`, the resulting is a model that has input of shape
        (batch, in_channels, width, height), and outputs an image of
        shape (batch, out_channels, width//pool_size, height//pool_size) 
        Inputs:
            in_channels: number of input channels
            out_channels: number of output channels
            depth: number of convolutional layers in encoder block
            kernel_size: size of the kernel of the convolutional layers
            pool_size: size of the kernel of the pooling layer
        """
        super(Encoder, self).__init__()

        # TODO: write code here

        # # of convolutions determined by depth 
        # 1 maxpool2d
                
        self.conv_layers = nn.ModuleList()
        for i in range(depth):
            # the first iteration is a special case where we increase # of channels 
            if i == 0:
                self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            else:
                self.conv_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2))
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, img):
        """
        runs the partial prediction in the encoder block

        Inputs:
            img: input image of shape
            (batch, in_channels, width, height)

        Outputs:
            img: output image of shape
            (batch, out_channels, width//pool_size, height//pool_size)
        """
        print(f"{img.size=}")
        for conv_layer in self.conv_layers:
            img = conv_layer(img)
            img = torch.relu(img)
        img = self.pool(img)
        img = torch.relu(img)
        return img

class SegmentationCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,  depth: int = 2, embedding_size: int = 64,
                 pool_sizes: List[int] = [5,5,2], kernel_size: int = 3, **kwargs):
        """
        Basic CNN that performs segmentation. This model takes
        an image of shape (batch, in_channels, width, height)
        and outputs an image of shape 
        (batch, out_channels, width//prod(pool_sizes), height//prod(pool_sizes)),
        where prod() is the product of all the values in pool_sizes.

        This is done by using len(pool_sizes) Encoder layers, each of which
        pools the resolution down by a factor of pool_sizes[i].

        The first encoder must project the in_channels to embedding_size.
        Each subsequent layer must double the number of channels in its input,
        for example, the second layer must go from embedding_size to 2*embedding_size,
        the third layer from 2*embedding_size to 4*embedding_size and so on, until
        the pool_sizes list has been depleted. 

        The final layer (decoder) must project the (2**len(pool_sizes))*embedding_size 
        channels to output_channels channels. In order to keep the resolution, you 
        may use a 1x1 kernel.

        Hint: If you use a regular list to save your Encoders, these
        will not register with the pytorch module and will not
        have their parameters added to the optimizer. Use nn.ModuleList to avoid this.

        Note: **kwargs is used to capture any additional arguments that are not
        explicitly defined in the function signature. This is so that we can
        construct the object from the same dictionary--certain models don't have addiitonal
        parameters.

        Inputs:
            in_channels: number of input channels
            out_channels: number of output channels
            depth: number of convolutional layers in encoder block
            embedding_size: number of channels in the first encoder
            pool_sizes: list of pool sizes for each encoder
            kernel_size: size of the kernel of the convolutional layers
        """
        super(SegmentationCNN, self).__init__()
        self.embedding_size = embedding_size
        self.encoder_list = nn.ModuleList()
        for i in range(len(pool_sizes)):
            if i == 0:
                self.encoder_list.append(Encoder(in_channels, embedding_size, depth, kernel_size, pool_sizes[i]))
            else:
                self.encoder_list.append(Encoder(self.embedding_size, self.embedding_size*2, depth, kernel_size, pool_sizes[i]))
                self.embedding_size = self.embedding_size*2
        self.decoder = nn.Conv2d(self.embedding_size, out_channels, 1)
        

    def forward(self, X):
        """
        Runs the input X through the encoders and decoders.
        Inputs:
            X: image of shape 
            (batch, in_channels, width, height)
        Outputs:
            y_pred: image of shape
            (batch, out_channels, width//prod(pool_sizes), height//prod(pool_sizes))
        """
        y_pred = X
        
        # run through encoders
        for encoder in self.encoder_list:
            y_pred = encoder(y_pred)
        # decode
        y_pred = self.decoder(y_pred)
        y_pred = torch.relu(y_pred)
        return y_pred