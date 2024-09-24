import torch
import torch.nn as nn

class EnergyBasedModel(nn.Module):
    def __init__(self, input_channel, batch_size, device):
        super(EnergyBasedModel, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.input_channel = input_channel
        self.net = nn.Sequential(
            nn.Linear(input_channel, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
    
    def sample_by_langevin(self, init_x, max_step=100, epsilon=10):
        x = init_x
        # x = torch.randn_like(init_x)
        x.requires_grad = True
        for i in range(max_step):
            noise = torch.randn_like(x)/255.
            energy = self.net(x).sum()
            grad = torch.autograd.grad(outputs=energy, inputs=x)[0]
            # if grad.max() > 10: print(f"grad: {grad.max()}")
            x = x + epsilon*grad + torch.sqrt(torch.tensor(2.*epsilon))*noise
            x = x.clip(0., 1.)
        return x

    def forward(self, x):
        # shape of x: (bs, c, h, w)
        batch_size, input_channel, height, weight = x.shape
        x = x.reshape(batch_size, -1)
        energy_real = self.net(x).squeeze()
        energy_langevin = self.net(self.sample_by_langevin(x)).squeeze()

        return energy_real, energy_langevin
