import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, filters):
        super(Generator, self).__init__()
        self.convolution_1 = nn.Conv2d(input_channels, filters, 6, 2, 2)
        self.convolution_2 = nn.Conv2d(filters, filters * 2, 6, 2, 2)
        self.convolution_3 = nn.Conv2d(filters * 2, filters * 4, 6, 2, 2)
        self.convolution_4 = nn.Conv2d(filters * 4, filters * 8, 6, 2, 2)
        self.convolution_5 = nn.Conv2d(filters * 8, filters * 8, 6, 2, 2)
        self.convolution_6 = nn.Conv2d(filters * 8, filters * 8, 6, 2, 2)
        self.convolution_7 = nn.Conv2d(filters * 8, filters * 8, 6, 2, 2)
        self.convolution_8 = nn.Conv2d(filters * 8, filters * 8, 6, 2, 2)

        self.deconvolution_1 = nn.ConvTranspose2d(filters * 8, filters * 8, 6, 2, 2)
        self.deconvolution_2 = nn.ConvTranspose2d(filters * 8 * 2, filters * 8, 6, 2, 2)
        self.deconvolution_3 = nn.ConvTranspose2d(filters * 8 * 2, filters * 8, 6, 2, 2)
        self.deconvolution_4 = nn.ConvTranspose2d(filters * 8 * 2, filters * 8, 6, 2, 2)
        self.deconvolution_5 = nn.ConvTranspose2d(filters * 8 * 2, filters * 4, 6, 2, 2)
        self.deconvolution_6 = nn.ConvTranspose2d(filters * 4 * 2, filters * 2, 6, 2, 2)
        self.deconvolution_7 = nn.ConvTranspose2d(filters * 2 * 2, filters, 6, 2, 2)
        self.deconvolution_8 = nn.ConvTranspose2d(filters * 2, output_channels, 6, 2, 2)

        self.batch_normalisation = nn.BatchNorm2d(filters)
        self.batch_normalisation_2 = nn.BatchNorm2d(filters * 2)
        self.batch_normalisation_4 = nn.BatchNorm2d(filters * 4)
        self.batch_normalisation_8 = nn.BatchNorm2d(filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.dropout = nn.Dropout(0.5)

        self.tanh = nn.Tanh()

    def forward(self, input):
        encoder_1 = self.convolution_1(input)
        encoder_2 = self.batch_normalisation_2(self.convolution_2(self.leaky_relu(encoder_1)))
        encoder_3 = self.batch_normalisation_4(self.convolution_3(self.leaky_relu(encoder_2)))
        encoder_4 = self.batch_normalisation_8(self.convolution_4(self.leaky_relu(encoder_3)))
        encoder_5 = self.batch_normalisation_8(self.convolution_5(self.leaky_relu(encoder_4)))
        encoder_6 = self.batch_normalisation_8(self.convolution_6(self.leaky_relu(encoder_5)))
        encoder_7 = self.batch_normalisation_8(self.convolution_7(self.leaky_relu(encoder_6)))
        encoder_8 = self.convolution_8(self.leaky_relu(encoder_7))

        decoder_1 = self.dropout(self.batch_normalisation_8(self.deconvolution_1(self.relu(encoder_8))))
        decoder_1 = torch.cat((decoder_1, encoder_7), 1)
        decoder_2 = self.dropout(self.batch_normalisation_8(self.deconvolution_2(self.relu(decoder_1))))
        decoder_2 = torch.cat((decoder_2, encoder_6), 1)
        decoder_3 = self.dropout(self.batch_normalisation_8(self.deconvolution_3(self.relu(decoder_2))))
        decoder_3 = torch.cat((decoder_3, encoder_5), 1)
        decoder_4 = self.batch_normalisation_8(self.deconvolution_4(self.relu(decoder_3)))
        decoder_4 = torch.cat((decoder_4, encoder_4), 1)
        decoder_5 = self.batch_normalisation_4(self.deconvolution_5(self.relu(decoder_4)))
        decoder_5 = torch.cat((decoder_5, encoder_3), 1)
        decoder_6 = self.batch_normalisation_2(self.deconvolution_6(self.relu(decoder_5)))
        decoder_6 = torch.cat((decoder_6, encoder_2),1)
        decoder_7 = self.batch_normalisation(self.deconvolution_7(self.relu(decoder_6)))
        decoder_7 = torch.cat((decoder_7, encoder_1), 1)
        decoder_8 = self.deconvolution_8(self.relu(decoder_7))
        output = self.tanh(decoder_8)
        return output

class Discriminator(nn.Module):
    def __init__(self, input_channels, output_channels, filters):
        super(Discriminator, self).__init__()
        self.convolution_1 = nn.Conv2d(input_channels + output_channels, filters, 6, 2, 1)
        self.convolution_2 = nn.Conv2d(filters, filters * 2, 6, 2, 1)
        self.convolution_3 = nn.Conv2d(filters * 2, filters * 4, 6, 2, 1)
        self.convolution_4 = nn.Conv2d(filters * 4, filters * 8, 6, 1, 1)
        self.convolution_5 = nn.Conv2d(filters * 8, 1, 6, 1, 1)

        self.batch_normalisation_2 = nn.BatchNorm2d(filters * 2)
        self.batch_normalisation_4 = nn.BatchNorm2d(filters * 4)
        self.batch_normalisation_8 = nn.BatchNorm2d(filters * 8)

        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        encoder_1 = self.convolution_1(input)
        encoder_2 = self.batch_normalisation_2(self.convolution_2(self.leaky_relu(encoder_1)))
        encoder_3 = self.batch_normalisation_4(self.convolution_3(self.leaky_relu(encoder_2)))
        encoder_4 = self.batch_normalisation_8(self.convolution_4(self.leaky_relu(encoder_3)))
        encoder_5 = self.convolution_5(self.leaky_relu(encoder_4))
        output = self.sigmoid(encoder_5)

        return output