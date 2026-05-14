## Implemeting a Siamese U-Net architecture
# Creating an Encoder for both SAR and EO streams. This forces the model to project both 
# modalities into a common feature space before comparing them

import torch
import torch.nn as nn
import torch.nn.functional as Func
from torchvision import models

class MobileNetV2Encoder(nn.Module):
    ##Encoder using MobileNetV2 features.
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
        ##We are using ImageNet pretrained weights as a starting point for our encoder. This allows the model to leverage learned features from a large dataset, which can be beneficial for our change detection task.
        self.enc0 = backbone[0:2] ## 112x112
        self.enc1 = backbone[2:4] ## 56x56
        self.enc2 = backbone[4:7] ## 28x28
        self.enc3 = backbone[7:14] ## 14x14
        self.enc4 = backbone[14:18] ## 7x7-> bottleneck
    
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1) ## Repeat 1-ch(Gray) to 3-ch(RGB) to satisfy MobileNetV2 requirements
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        out = self.enc4(s3)
        return out, [s3, s2, s1, s0]

## Decoder that fuses features from prev layers and TWO skip connections(EO+SAR)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch*2, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_eo, skip_sar):
        x = self.up(x)
        if x.shape[2:] != skip_eo.shape[2:]: ##Handles cases where feature map sizes differ by 1 pixel
            x = Func.interpolate(x, size=skip_eo.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip_eo, skip_sar], dim=1)
        return self.conv(x)

class SiameseUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.encoder = MobileNetV2Encoder()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(320 * 2, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        ##Setting decoder path: Channel sizes corresponding to MobileNetV2 skipping outputs
        self.dec4 = DecoderBlock(256, 96, 128)
        self.dec3 = DecoderBlock(128, 32, 64)
        self.dec2 = DecoderBlock(64, 24, 32)
        self.dec1 = DecoderBlock(32, 16, 16)
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1) ##Output single channel binary mask
    
    def forward(self, eo, sar):
        f_eo, skips_eo = self.encoder(eo)
        f_sar, skips_sar = self.encoder(sar)

        ##Deep fusion at bottleneck
        b = self.bottleneck(torch.cat([f_eo, f_sar], dim=1))

        ## Progressive decoding, where we reconstruct the binary mask
        y = self.dec4(b, skips_eo[0], skips_sar[0])
        y = self.dec3(y, skips_eo[1], skips_sar[1])
        y = self.dec2(y, skips_eo[2], skips_sar[2])
        y = self.dec1(y, skips_eo[3], skips_sar[3]) 
        out = self.final_conv(y)
        return Func.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)