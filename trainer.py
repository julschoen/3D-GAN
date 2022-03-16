import os
import numpy as np
import pytorch_fid_wrapper as FID

import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.utils as vutils

from model import Discriminator, Generator


class Params(object):
    def __init__(self, **kwargs):
        self.lrG = 0.0002
        self.lrD = 0.0002
        self.beta1G = 0.5
        self.beta1D = 0.5
        self.epochs = 50
        self.batch_size = 32
        self.ngpu=2

        self.steps_per_log = 10
        self.steps_per_img_log = 50

        ### Model Params ###
        self.z_size = 100
        self.filterG = 256
        self.filterD = 128

        self.clamp_lower = -0.01
        self.clamp_upper = 0.01

        for key, val in kwargs.items():
            if val is not None:
                self.__dict__[key] = val

class Trainer(object):
    def __init__(self, dataset, params=Params(), log_dir='', device='cuda'):
        ### Misc ###
        self.p = params
        self.device = device

        ### Make Dirs ###
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.models_dir = os.path.join(self.log_dir, 'models')
        self.images_dir = os.path.join(self.log_dir, 'images')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        ### Make Models ###
        self.netG = Generator(self.p).to(device)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.p.lrG,
                                     betas=(self.p.beta1G, 0.999))
        self.netD = Discriminator(self.p).to(device)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.p.lrD,
                                     betas=(self.p.beta1D, 0.999))
        self.netG.apply(self.weights_init)
        self.netD.apply(self.weights_init)

        ### Make Data Generator ###
        self.generator_train = DataLoader(dataset, batch_size=self.p.batch_size, shuffle=True, num_workers=2)

        ### Prep Training
        self.fixed_test_noise = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.fid = []
        self.fid_epoch = []

    def make_labels(self, size):
        labels = (torch.randint(900,1000, size, device=self.device)/1000).float()
        i = torch.randint(0,100, size)
        labels[i < 5] = 0.
        return labels

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        
    def log_train(self, epoch, step, fake, real, D_x, D_G_z1, D_G_z2):
        with torch.no_grad():
            self.fid.append(
                FID.fid(
                    torch.reshape(fake, (-1,1,128,128)).expand(-1,3,-1,-1), 
                    real_images=torch.reshape(real, (-1,1,128,128)).expand(-1,3,-1,-1)
                    )
                )

        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tFID %.4f'
                    % (epoch+1, self.p.epochs, step%len(self.generator_train), len(self.generator_train),
                        self.D_losses[-1], self.G_losses[-1], D_x, D_G_z1, D_G_z2, self.fid[-1]))

    def log_interpolation(self, step):
        noise = torch.randn(self.p.batch_size, self.p.z_size, 1, 1,1,
                            dtype=torch.float, device=self.device)
        if self.fixed_test_noise is None:
            self.fixed_test_noise = noise.clone()
    
        with torch.no_grad():
            fake = self.netG(self.fixed_test_noise).detach().cpu()
        torchvision.utils.save_image(
            vutils.make_grid(torch.reshape(fake, (-1,1,128,128)), padding=2, normalize=True)
            , os.path.join(self.images_dir, f'{step}.png'))

    def start_from_checkpoint(self):
        step = 0
        epoch_done = 0
        checkpoint = os.path.join(self.models_dir, 'checkpoint.pt')
        if os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint)
            step = state_dict['step']
            epoch_done = state_dict['epoch'] +1
            self.netG.load_state_dict(state_dict['modelG_state_dict'])
            self.netD.load_state_dict(state_dict['modelD_state_dict'])

            self.optimizerG.load_state_dict(state_dict['optimizerG_state_dict'])
            self.optimizerD.load_state_dict(state_dict['optimizerD_state_dict'])

            self.G_losses = state_dict['lossG']
            self.D_losses = state_dict['lossD']
            self.fid_epoch = state_dict['fid']
            print('starting from step {}'.format(step))
        return step, epoch_done

    def save_checkpoint(self, epoch, step):
        torch.save({
        'epoch': epoch,
        'step': step,
        'modelG_state_dict': self.netG.state_dict(),
        'modelD_state_dict': self.netD.state_dict(),
        'optimizerG_state_dict': self.optimizerG.state_dict(),
        'optimizerD_state_dict': self.optimizerD.state_dict(),
        'lossG': self.G_losses,
        'lossD': self.D_losses,
        'fid': self.fid_epoch,
        }, os.path.join(self.models_dir, 'checkpoint.pt'))

    def log(self, epoch, step, fake, real, D_x, D_G_z1, D_G_z2):
        if step % self.p.steps_per_log == 0:
            self.log_train(epoch, step, fake, real, D_x, D_G_z1, D_G_z2)

        if step % self.p.steps_per_img_log == 0:
            self.log_interpolation(step)

    def log_final(self, epoch, step, fake, real, D_x, D_G_z1, D_G_z2):
        self.log_train(epoch, step, fake, real, D_x, D_G_z1, D_G_z2)
        self.log_interpolation(step)
        self.save_checkpoint(epoch, step)

    def train(self):
        step, epoch_done = self.start_from_checkpoint()
        FID.set_config(device=self.device)

        print("Starting Training...")
        for epoch in range(epoch_done, self.p.epochs):
            for i, data in enumerate(self.generator_train, 0):      
                
                for p in self.netD.parameters():
                    p.requires_grad = True

                for p in self.netD.parameters():
                    p.data.clamp_(self.p.clamp_lower, self.p.clamp_upper)

                real = data.to(self.device).unsqueeze(dim=1)

                label = self.make_labels((self.p.batch_size,))
                
                self.netD.zero_grad()
                errD_real = self.netD(real)
                errD_real.backward()

                noise = torch.randn(self.p.batch_size, self.p.z_size, 1, 1,1,
                                    dtype=torch.float, device=self.device)
                fake = self.netG(noise)
                label = label.fill_(0.)

                errD_fake = self.netD(fake.detach())
                errD_fake.backward()
                errD = errD_real - errD_fake

                self.optimizerD.step()

                for p in self.netD.parameters():
                    p.requires_grad = False

                self.netG.zero_grad()
                label = label.fill_(1.)  # fake labels are real for generator cost
                
                errG = self.netD(fake)
                errG.backward()

                self.optimizerG.step()
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())

                self.log(epoch, step, fake, real, errD_real.item(), errD_fake.item(), errG.item())

                step += 1
            
            self.fid_epoch.append(np.array(self.fid).mean())
            self.fid = []
            self.save_checkpoint(epoch, step)
        
        self.log_final(epoch, step, fake, real, D_x.item(), D_G_z1.item(), D_G_z2.item())
        print('...Done')