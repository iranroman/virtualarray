import torch.nn as nn

# in_channels = 16 <- number of PQMF bands

class Encoder(nn.Module):

    # inspired by:
    #     https://github.com/acids-ircam/RAVE/blob/
    #     54c6106eca9760041e7d01e10ba1e9f51f04fc5a/
    #     rave/model.py#L294

    # added coordinate projections from:
    #     https://github.com/vivjay30/Cone-of-Silence/blob/
    #     f84f4168ab822734856af7574b951bd2e37c9e2f/cos/
    #     training/network.py#L158

    def __init__(
                self, 
                in_channels,
                kernel_size,
                capacity = 64,
                ratios = [4, 4, 4, 2],
                padding = 'same',
                bias = True
                ):
        super().__init__()

        # operations in first convolution block
        self.conv1 = nn.Conv1d(
                in_channels = in_channels,
                out_channels = capacity,
                kernel_size = kernel_size,
                padding = padding,
                bias = bias)
        self.bn1 = nn.BatchNorm1d(capacity))
        self.lr1 = nn.LeakyReLU(0.2)
        self.coordconv1 = nn.Conv1d(3,capacity,1)

        # operations in second convolution block
        in_dim = 2**0 * capacity
        out_dim = 2**(0 + 1) * capacity
        self.conv2 = nn.Conv1d(
                in_channels = in_dim,
                out_channels = out_dim,
                kernel_size = 2 * r[0] + 1,
                padding = padding,
                stride = r[0],
                bias = bias)
        self.bn2 = nn.BatchNorm1d(out_dim))
        self.lr2 = nn.LeakyReLU(0.2)
        self.coordconv2 = nn.Conv1d(3,out_dim,1)

        # operations in third convolution block
        in_dim = 2**1 * capacity
        out_dim = 2**(1 + 1) * capacity
        self.conv3 = nn.Conv1d(
                in_channels = in_dim,
                out_channels = out_dim,
                kernel_size = 2 * r[1] + 1,
                padding = padding,
                stride = r[1],
                bias = bias)
        self.bn3 = nn.BatchNorm1d(out_dim))
        self.lr3 = nn.LeakyReLU(0.2)
        self.coordconv3 = nn.Conv1d(3,out_dim,1)

        # operations in fourth convolution block
        in_dim = 2**2 * capacity
        out_dim = 2**(2 + 1) * capacity
        self.conv4 = nn.Conv1d(
                in_channels = in_dim,
                out_channels = out_dim,
                kernel_size = 2 * r[2] + 1,
                padding = padding,
                stride = r[2],
                bias = bias)
        self.lr4 = nn.LeakyReLU(0.2)
        self.coordconv4 = nn.Conv1d(3,out_dim,1)

        # operations in fifth convolution block
        self.conv5 = nn.Conv1d(
                in_channels = out_dim,
                out_channels = latent_size,
                kernel_size = kernel_size - 2,
                padding = padding,
                bias = bias)
        self.coordconv5 = nn.Conv1d(3,latent_size,1)

    def forward(self, x):
        return z