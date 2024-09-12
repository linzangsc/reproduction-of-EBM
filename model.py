import torch
import torch.nn as nn

class EnergyBasedModel(nn.Module):
    def __init__(self, input_channel, batch_size, device):
        super(EnergyBasedModel, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.input_channel = input_channel
        self.net = nn.Sequential(
            nn.Linear(input_channel, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def sample_by_langevin(self, init_x, max_step=10, epsilon=1e-3):
        x = init_x
        x.requires_grad = True
        noise = torch.randn_like(x)
        for i in range(max_step):
            energy = self.calculate_energy(x).sum()
            grad = torch.autograd.grad(outputs=energy, inputs=x, create_graph=True)[0]
            x = x + epsilon*grad + torch.sqrt(torch.tensor(2.*epsilon))*noise
        return x

    def calculate_energy(self, x):
        return torch.exp(self.net(x).squeeze())

    def forward(self, x):
        # shape of x: (bs, c, h, w)
        batch_size, input_channel, height, weight = x.shape
        x = x.reshape(batch_size, -1)
        energy_real = self.calculate_energy(x)
        energy_model = self.calculate_energy(self.sample_by_langevin(x))

        return energy_real, energy_model
