import os
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import CustomizedDataset, visualize_float_result, visualize_latent_space, Sampler
from model import EnergyBasedModel
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal

class Trainer:
    def __init__(self, config) -> None:
        self.config = config
        self.image_size = config['image_size']
        self.logger = SummaryWriter(self.config['log_path'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = config['batch_size']
        self.num_epochs = config['num_epochs']
        self.dataset = CustomizedDataset()
        self.train_dataset = self.dataset.train_dataset
        self.test_dataset = self.dataset.test_dataset
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, 
                                                        batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset, 
                                                       batch_size=self.batch_size, shuffle=False)
        input_channel = self.image_size*self.image_size
        self.model = EnergyBasedModel(input_channel=input_channel, batch_size=self.batch_size,
                                      device=self.device).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'], betas=(0.0, 0.999))
        self.sampler = Sampler(self.model, img_shape=(1, self.image_size, self.image_size), 
                               sample_size=self.batch_size, device=self.device)

    def loss(self, energy_real, energy_langevin):
        cd_loss = (energy_langevin - energy_real).mean()
        reg_loss = (energy_real**2 + energy_langevin**2).mean()
        return cd_loss, reg_loss

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                # translate to binary images
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                fake_images = self.sampler.sample_new_exmps(steps=60, step_size=10)
                # fake_images = self.sampler.generate_samples(self.model, images.clone(), steps=100, step_size=10)
                cat_images = torch.cat([images, fake_images], dim=0)
                energy_real, energy_langevin = self.model(cat_images).chunk(2, dim=0)
                cd_loss, reg_loss = self.loss(energy_real, energy_langevin)
                loss = cd_loss + 0.1*reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                self.optimizer.step()
                if i % 1 == 0:
                    # from IPython import embed; embed()
                    print(f'Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{len(self.train_loader)}], loss: {loss.item():.6f}, \
                           cd_loss:{cd_loss.item():.6f}, reg_loss:{reg_loss.item():.6f}, real_energy: {-energy_real.mean():.6f}, fake_energy: {-energy_langevin.mean():.6f}')
                self.logger.add_scalar('loss/train', loss.item(), i + epoch * len(self.train_loader))
                self.logger.add_scalar('loss/cd', cd_loss.item(), i + epoch * len(self.train_loader))
                self.logger.add_scalar('loss/reg', reg_loss.item(), i + epoch * len(self.train_loader))
                self.logger.add_scalar('energy/real', -energy_real.mean(), i + epoch * len(self.train_loader))
                self.logger.add_scalar('energy/fake', -energy_langevin.mean(), i + epoch * len(self.train_loader))
                
            self.save_model(self.config['ckpt_path'])

            z = torch.rand((16, 1, self.image_size, self.image_size)).to(self.device)
            z = z * 2 - 1
            sample_image = self.sampler.generate_samples(self.model, z, steps=256, step_size=10)
            sample_image = sample_image.reshape(16, 1, self.image_size, self.image_size)
            self.visualize_samples(sample_image, epoch)

    def save_model(self, output_path):
        if not os.path.exists(output_path): os.mkdir(output_path)
        torch.save(self.model.state_dict(), os.path.join(output_path, f"model.pth"))

    def visualize_samples(self, sample_images, epoch):
        sample_images = sample_images.reshape(sample_images.shape[0], self.image_size, self.image_size).detach().to('cpu')
        npy_sampled_theta = np.array(sample_images)
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        axs = visualize_float_result(npy_sampled_theta, axs)
        self.logger.add_figure(f"sample results", plt.gcf(), epoch)
        plt.close(fig)

    def visualize_latent_space(self, epoch):
        fig, ax = plt.subplots()
        for (images, labels) in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, _, _, latents = self.model(images)
            ax = visualize_latent_space(latents, labels, ax)
        plt.colorbar(ax.collections[0], ax=ax)
        self.logger.add_figure(f"latent space", plt.gcf(), epoch)
        plt.close(fig)

if __name__ == "__main__":
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    trainer = Trainer(config=config)
    trainer.train()
