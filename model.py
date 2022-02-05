import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, kernel_size=3):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channel, out_channel, 
            kernel_size=kernel_size, stride=stride, padding=1
        )
        self.act_func = nn.ReLU()
        self.pooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.pooling(self.act_func(self.conv(x)))


class DeConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, kernel_size=3):
        super().__init__()
        
        self.deconv = nn.ConvTranspose2d(
            in_channel, out_channel, 
            kernel_size=kernel_size, stride=stride, padding=1
        )
        self.act_func = nn.ReLU()
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        return self.upsampling(self.act_func(self.deconv(x)))

        
class ConvAutoencoder(nn.Module):
    """ Convolutional Autoencoder for Mel-Spectrogram 
    (N, 1, 48, 48)
    #TODO: 필요하다면 conv layer parameter argument로 할 수 있도록 변경
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvLayer(1, 32),
            ConvLayer(32, 64),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 1024),
            nn.Linear(1024, 64)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 1024),
            nn.Linear(1024, 64 * 12 * 12),
            nn.Unflatten(1, (64, 12, 12)),
            DeConvLayer(64, 64),
            DeConvLayer(64, 32),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)

        return out, latent



### Deprecated

# class ConvAutoEncoder(nn.Module):
#     """ Convolutional Autoencoder for Mel spectrogram """
#     def __init__(self, channels=[1, 8, 16, 16]):
#         super().__init__()

#         encoder_layers = []
#         for in_channel, out_channel in zip(channels, channels[1:]):
#             encoder_layers.extend([
#                 nn.Conv2d(in_channel, out_channel, kernel_size=3),
#                 nn.ReLU(),
#                 nn.MaxPool2d(kernel_size=2)
#             ])
#         encoder_layers.extend([
#             nn.Flatten(),
#             nn.Linear(in_features=16 * 4 * 232, out_features=4096),
#             nn.Linear(in_features=4096, out_features=512)
#         ])
#         self.encoder = nn.Sequential(*encoder_layers)
        
#         channels.append(channels[-1])
#         channels.reverse()

#         decoder_layers = []
#         decoder_layers.extend([
#             nn.Linear(in_features=512, out_features=4096),
#             nn.Linear(in_features=4096, out_features=16 * 4 * 232),
#             nn.Unflatten(1, (16, 4, 232)),
#         ])
#         for in_channel, out_channel in zip(channels[:-1], channels[1:]):
#             decoder_layers.extend([
#                 nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3),
#                 nn.ReLU(),
#                 nn.UpsamplingNearest2d(scale_factor=2),
#             ])
#         decoder_layers.extend([
#             nn.ConvTranspose2d(channels[-2], channels[-1], kernel_size=3),
#             nn.Sigmoid()
#         ])
#         self.decoder = nn.Sequential(*decoder_layers)

#     def forward(self, x):
#         """
#         Args:
#             x (np.array): expected shape: (batch_size, 1, 48, 1876)
#         """
#         latent = self.encoder(x)
#         decoded = self.decoder(latent)

#         return latent, decoded
