import torch.nn as nn


class ConvAutoEncoder(nn.Module):
    """ Convolutional Autoencoder for Mel spectrogram """
    def __init__(self, channels=[1, 8, 16, 16]):
        super().__init__()

        encoder_layers = []
        for in_channel, out_channel in zip(channels, channels[1:]):
            encoder_layers.extend([
                nn.Conv2d(in_channel, out_channel, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            ])
        encoder_layers.extend([
            nn.Flatten(),
            nn.Linear(in_features=16 * 4 * 232, out_features=4096),
            nn.Linear(in_features=4096, out_features=512)
        ])
        self.encoder = nn.Sequential(*encoder_layers)
        
        channels.append(channels[-1])
        channels.reverse()

        decoder_layers = []
        decoder_layers.extend([
            nn.Linear(in_features=512, out_features=4096),
            nn.Linear(in_features=4096, out_features=16 * 4 * 232),
            nn.Unflatten(1, (16, 4, 232)),
        ])
        for in_channel, out_channel in zip(channels[:-1], channels[1:]):
            decoder_layers.extend([
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3),
                nn.ReLU(),
                nn.UpsamplingNearest2d(scale_factor=2),
            ])
        decoder_layers.extend([
            nn.ConvTranspose2d(channels[-2], channels[-1], kernel_size=3),
            nn.Sigmoid()
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Args:
            x (np.array): expected shape: (batch_size, 1, 48, 1876)
        """
        latent = self.encoder(x)
        decoded = self.decoder(latent)

        return latent, decoded
