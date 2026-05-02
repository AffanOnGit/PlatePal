import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, embed_dim=512, out_channels=3):
        super(Generator, self).__init__()
        # Input: concatenated noise (z) and CLIP embedding
        self.init_size = 4
        self.l1 = nn.Sequential(nn.Linear(z_dim + embed_dim, 512 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, out_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, embedding):
        # Concatenate noise and embedding
        x = torch.cat((z, embedding), dim=1)
        out = self.l1(x)
        out = out.view(out.shape[0], 512, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 64 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

if __name__ == "__main__":
    # Test shapes
    z = torch.randn(1, 100)
    embed = torch.randn(1, 512)
    gen = Generator()
    img = gen(z, embed)
    print(f"Generator output shape: {img.shape}") # Should be [1, 3, 64, 64]
    
    disc = Discriminator()
    validity = disc(img)
    print(f"Discriminator output shape: {validity.shape}") # Should be [1, 1]
