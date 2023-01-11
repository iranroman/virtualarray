import torch.nn as nn

# in_channels = 16 <- number of PQMF bands

# should we use weight norm?

class ResidualStack(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size,
                 padding,
                 bias=False): # try also with bias (?)
        super().__init__()
        net = []

        # SEQUENTIAL RESIDUALS
        for i in range(3):
            # RESIDUAL BLOCK
            seq = [nn.LeakyReLU(.2)]
            seq.append(
                nn.Conv1d(
                     dim,
                     dim,
                     kernel_size,
                     padding=padding,
                     dilation=3**i,
                     bias=bias,
                ))

            seq.append(nn.LeakyReLU(.2))
            seq.append(
                nn.Conv1d(
                    dim,
                    dim,
                    kernel_size,
                    padding=padding,
                    bias=bias,
                ))

            res_net = nn.Sequential(*seq)

            net.append(res_net)

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


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

        ###########
        # ENCODER #
        ###########

        # operations in first encoder layer
        self.conv1 = nn.Conv2d(
                in_channels = in_channels,
                out_channels = capacity,
                kernel_size = (1, kernel_size),
                padding = padding,
                bias = bias)
        self.bn1 = nn.BatchNorm1d(capacity))
        self.lr1 = nn.LeakyReLU(0.2)
        self.coordconv1 = nn.Conv2d(3,capacity,1)

        # operations in second encoder layer
        in_dim = 2**0 * capacity
        out_dim = 2**(0 + 1) * capacity
        self.conv2 = nn.Conv2d(
                in_channels = in_dim,
                out_channels = out_dim,
                kernel_size = (1, 2 * r[0] + 1),
                padding = padding,
                stride = (1, r[0]),
                bias = bias)
        self.bn2 = nn.BatchNorm1d(out_dim))
        self.lr2 = nn.LeakyReLU(0.2)
        self.coordconv2 = nn.Conv2d(3,out_dim,1)

        # operations in third encoder layer
        in_dim = 2**1 * capacity
        out_dim = 2**(1 + 1) * capacity
        self.conv3 = nn.Conv2d(
                in_channels = in_dim,
                out_channels = out_dim,
                kernel_size = (1, 2 * r[1] + 1),
                padding = padding,
                stride = (1, r[1]),
                bias = bias)
        self.bn3 = nn.BatchNorm1d(out_dim))
        self.lr3 = nn.LeakyReLU(0.2)
        self.coordconv3 = nn.Conv2d(3,out_dim,1)

        # operations in fourth encoder layer
        in_dim = 2**2 * capacity
        out_dim = 2**(2 + 1) * capacity
        self.conv4 = nn.Conv2d(
                in_channels = in_dim,
                out_channels = out_dim,
                kernel_size = (1, 2 * r[2] + 1),
                padding = padding,
                stride = (1, r[2]),
                bias = bias)
        self.bn4 = nn.BatchNorm1d(out_dim))
        self.lr4 = nn.LeakyReLU(0.2)
        self.coordconv4 = nn.Conv2d(3,out_dim,1)

        # operations in fourth encoder layer
        in_dim = 2**3 * capacity
        out_dim = 2**(3 + 1) * capacity
        self.conv5 = nn.Conv2d(
                in_channels = in_dim,
                out_channels = out_dim,
                kernel_size = (1, 2 * r[3] + 1),
                padding = padding,
                stride = (1, r[3]),
                bias = bias)
        self.bn5 = nn.BatchNorm1d(out_dim))
        self.lr5 = nn.LeakyReLU(0.2)
        self.coordconv5 = nn.Conv2d(3,out_dim,1)

        # operations in sixth encoder layer
        self.conv6 = nn.Conv2d(
                in_channels = out_dim,
                out_channels = latent_size,
                kernel_size = (1, kernel_size - 2),
                padding = padding,
                bias = bias)
        self.coordconv6 = nn.Conv2d(3,latent_size,1)

        ###########
        # DECODER #
        ###########

        # operations in first decoder layer
        out_dim = 2**(4) * capacity
        self.deconv1 = nn.Conv1d(
                in_channels = latent_size,
                out_channels = out_dim,
                kernel_size = kernel_size,
                padding = padding,
                bias = bias)
        self.coorddeconv1 = nn.Conv1d(3,latent_size,1)
        self.lrd1 = nn.LeakyReLU(0.2)

        # operations in the second decoder layer
        in_dim = 2**(4) * capacity
        out_dim = 2**(3) * capacity
        self.deconv2 = nn.ConvTranspose1d(
                in_channels = in_dim,
                out_channels = out_dim,
                kernel_size = 2 * r[0],
                stride = r[0],
                padding = r[0] // 2,
                bias = bias)
		self.resnet2 = ResidualStack(
				dim=out_dim,
				kernel_size=3,
				padding=padding)
        self.lrd2 = nn.LeakyReLU(0.2)

        # operations in the third decoder layer
        in_dim = 2**(3) * capacity
        out_dim = 2**(2) * capacity
        self.deconv2 = nn.ConvTranspose1d(
                in_channels = in_dim,
                out_channels = out_dim,
                kernel_size = 2 * r[1],
                stride = r[1],
                padding = r[1] // 2,
                bias = bias)
		self.resnet2 = ResidualStack(
				dim=out_dim,
				kernel_size=3,
				padding=padding)
        self.lrd2 = nn.LeakyReLU(0.2)

        # operations in the fourth decoder layer
        in_dim = 2**(2) * capacity
        out_dim = 2**(1) * capacity
        self.deconv2 = nn.ConvTranspose1d(
                in_channels = in_dim,
                out_channels = out_dim,
                kernel_size = 2 * r[2],
                stride = r[2],
                padding = r[2] // 2,
                bias = bias)
		self.resnet2 = ResidualStack(
				dim=out_dim,
				kernel_size=3,
				padding=padding)
        self.lrd2 = nn.LeakyReLU(0.2)

        # operations in the fifth decoder layer
        in_dim = 2**(1) * capacity
        out_dim = 2**(0) * capacity
        self.deconv2 = nn.ConvTranspose1d(
                in_channels = in_dim,
                out_channels = out_dim,
                kernel_size = 2 * r[3],
                stride = r[3],
                padding = r[3] // 2,
                bias = bias)
		self.resnet2 = ResidualStack(
				dim=out_dim,
				kernel_size=3,
				padding=padding)


    def forward(self, x, coords_in, coords_out):

        # first encoder layer
        x = self.conv1(x)
        x += self.coordconv1(coords)
        x = self.bn1(x)
        x = self.lr1(x)

        # second encoder layer
        x = self.conv2(x)
        x += self.coordconv2(coords)
        x = self.bn2(x)
        x = self.lr2(x)

        # third encoder layer
        x = self.conv3(x)
        x += self.coordconv3(coords)
        x = self.bn3(x)
        x = self.lr3(x)

        # fourth encoder layer
        x = self.conv4(x)
        x += self.coordconv4(coords)
        x = self.bn4(x)
        x = self.lr4(x)

        # fifth encoder layer
        x = self.conv5(x)
        x += self.coordconv5(coords)
        x = self.bn5(x)
        x = self.lr5(x)

        # sixth encoder layer
        x = self.conv6(x)
        x += self.coordconv6(coords)

        # average(?) over the coords dimension
        x = torch.mean(x,dim=2)

        # first decover layer

        return x
