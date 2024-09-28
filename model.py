import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class CNNModel(nn.Module):

    def __init__(self, hidden_features=32, out_dim=1, **kwargs):
        super().__init__()
        # We increase the hidden dimension over layers. Here pre-calculated for simplicity.
        c_hid1 = hidden_features//2
        c_hid2 = hidden_features
        c_hid3 = hidden_features*2
        
        # Series of convolutions and Swish activation functions
        self.cnn_layers = nn.Sequential(
                nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4), # [16x16] - Larger padding to get 32x32 image
                Swish(),
                nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1), #  [8x8]
                Swish(),
                nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1), # [4x4]
                Swish(),
                nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1), # [2x2]
                Swish(),
                nn.Flatten(),
                nn.Linear(c_hid3*4, c_hid3),
                Swish(),
                nn.Linear(c_hid3, out_dim)
        )

    def forward(self, x):
        x = self.cnn_layers(x).squeeze(dim=-1)
        return x

class EnergyBasedModel(nn.Module):
    def __init__(self, input_channel, batch_size, device):
        super(EnergyBasedModel, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.input_channel = input_channel
        self.net = CNNModel()
    
    def sample_by_langevin(self, init_x, max_step=100, epsilon=10):
        x = init_x
        # x = torch.randn_like(init_x)
        x.requires_grad = True
        noise = torch.randn_like(x)
        for i in range(max_step):
            noise.normal_(0, 0.005)
            energy = self.net(x).sum()
            grad = torch.autograd.grad(outputs=energy, inputs=x)[0]
            grad.clamp_(-0.03, 0.03)
            x = x + epsilon*grad + torch.sqrt(torch.tensor(2.*epsilon))*noise
            x = x.clip(-1., 1.)
        return x.detach()

    def forward(self, x):
        return self.net(x)
