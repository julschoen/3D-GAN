import os
import numpy as np
import pytorch_fid_wrapper as FID
import pickle
import os
from carbontracker.tracker import CarbonTracker

import torch 
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable, grad
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.utils as vutils
#from torchsummary import summary

from dcgan import Discriminator, Generator
from biggan import Discriminator as BigD
from biggan import Generator as BigG
from style_test import Discriminator as StyleD
from style_test import Generator as StyleG
from stylegan import EMA


class Trainer(object):
    def __init__(self, dataset, params):
        ### Misc ###
        self.device = params.device

        ### Make Dirs ###
        self.log_dir = params.log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.models_dir = os.path.join(self.log_dir, 'models')
        self.images_dir = os.path.join(self.log_dir, 'images')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)

        ### load/save params
        if params.load_params:
            with open(os.path.join(params.log_dir, 'params.pkl'), 'rb') as file:
                params = pickle.load(file)
        else:
            with open(os.path.join(params.log_dir,'params.pkl'), 'wb') as file:
                pickle.dump(params, file)

        self.p = params

        ### Make Models ###
        if self.p.hybrid:
            self.netG = BigG(self.p).to(self.device)
            self.netD = Discriminator(self.p).to(self.device)
        elif self.p.dcgan:
            self.netD = Discriminator(self.p).to(self.device)
            self.netG = Generator(self.p).to(self.device)
        elif self.p.stylegan:
            self.netD = StyleD(self.p).to(self.device)
            self.netG = StyleG(self.p).to(self.device)
            self.pl_mean = None
            self.pl_length_ema = EMA(0.99)
        else:
            self.netD = BigD(self.p).to(self.device)
            self.netG = BigG(self.p).to(self.device)

        #print('Summary G')
        #summary(self.netG, input_size=(params.z_size, 1, 1, 1))

        #print('Summary D')
        #summary(self.netD, input_size=(1, 128, 128, 128))

        if self.p.ngpu > 1:
            self.netD = nn.DataParallel(self.netD)
            self.netG = nn.DataParallel(self.netG)

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.p.lrD, betas=(0., 0.9))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.p.lrG, betas=(0., 0.9))

        self.scalerD = GradScaler()
        self.scalerG = GradScaler()

        ### Make Data Generator ###
        self.generator_train = DataLoader(dataset, batch_size=self.p.batch_size, shuffle=True, num_workers=4, drop_last=True)

        ### Prep Training
        self.fixed_test_noise = None
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.fid = []
        self.fid_epoch = []
        #self.tracker = CarbonTracker(epochs=self.p.niters, log_dir=self.p.log_dir)

    def inf_train_gen(self):
        while True:
            for data in self.generator_train:
                yield data
        
    def log_train(self, step, fake, real):
        with torch.no_grad():
            self.fid.append(
                FID.fid(
                    torch.reshape(fake.to(torch.float32), (-1,1,128,128)).expand(-1,3,-1,-1), 
                    real_images=torch.reshape(real.to(torch.float32), (-1,1,128,128)).expand(-1,3,-1,-1)
                    )
                )
        d_real, d_fake = self.D_losses[-1]
        print('[%d|%d]\tD(x): %.4f\tD(G(z)): %.4f|%.4f\tFID %.4f'
                    % (step, self.p.niters, d_real, d_fake, self.G_losses[-1], self.fid[-1]))

    def log_interpolation(self, step):
        noise = torch.randn(self.p.batch_size, self.p.z_size, 1, 1,1,
                            dtype=torch.float, device=self.device)
        if self.fixed_test_noise is None:
            self.fixed_test_noise = noise.clone()
    
        with torch.no_grad():
            if self.p.stylegan:
                fake, _ = self.netG(self.fixed_test_noise)
                fake = fake.detach().cpu()
            else:
                fake = self.netG(self.fixed_test_noise).detach().cpu()
        torchvision.utils.save_image(
            vutils.make_grid(torch.reshape(fake, (-1,1,128,128)), padding=2, normalize=True)
            , os.path.join(self.images_dir, f'{step}.png'))

    def start_from_checkpoint(self):
        step = 0
        checkpoint = os.path.join(self.models_dir, 'checkpoint.pt')
        if os.path.isfile(checkpoint):
            state_dict = torch.load(checkpoint)
            step = state_dict['step']

            self.optimizerG.load_state_dict(state_dict['optimizerG_state_dict'])
            self.optimizerD.load_state_dict(state_dict['optimizerD_state_dict'])
            
            
            self.netG.load_state_dict(state_dict['modelG_state_dict'])
            self.netD.load_state_dict(state_dict['modelD_state_dict'])

            self.optimizerG.load_state_dict(state_dict['optimizerG_state_dict'])
            self.optimizerD.load_state_dict(state_dict['optimizerD_state_dict'])

            self.G_losses = state_dict['lossG']
            self.D_losses = state_dict['lossD']
            self.fid_epoch = state_dict['fid']
            print('starting from step {}'.format(step))
        return step

    def save_checkpoint(self, step):
        torch.save({
        'step': step,
        'modelG_state_dict': self.netG.state_dict(),
        'modelD_state_dict': self.netD.state_dict(),
        'optimizerG_state_dict': self.optimizerG.state_dict(),
        'optimizerD_state_dict': self.optimizerD.state_dict(),
        'lossG': self.G_losses,
        'lossD': self.D_losses,
        'fid': self.fid_epoch,
        }, os.path.join(self.models_dir, 'checkpoint.pt'))

    def log(self, step, fake, real):
        if step % self.p.steps_per_log == 0:
            self.log_train(step, fake, real)

        if step % self.p.steps_per_img_log == 0:
            self.log_interpolation(step)

    def log_final(self, step, fake, real):
        self.log_train(step, fake, real)
        self.log_interpolation(step)
        self.save_checkpoint(step)

    def calc_gradient_penalty(self, real_data, fake_data):
        with autocast():
            alpha = torch.rand(real_data.shape[0], 1, 1, 1, 1)
            alpha = alpha.expand_as(real_data)
            alpha = alpha.to(self.device)
            interpolates = alpha * real_data + ((1 - alpha) * fake_data)

            
            interpolates = interpolates.to(self.device)
            interpolates = Variable(interpolates, requires_grad=True)
            disc_interpolates = self.netD(interpolates)
            
            gradients = grad(outputs=disc_interpolates,
                             inputs=interpolates,
                             grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                             retain_graph=True)[0]

            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) **2).mean() * 10
        return gradient_penalty

    def train(self):
        step_done = self.start_from_checkpoint()
        FID.set_config(device=self.device)
        one = torch.FloatTensor([1]).to(self.device)
        mone = one * -1
        gen = self.inf_train_gen()
        for p in self.netD.parameters():
                p.requires_grad = False
        for p in self.netG.parameters():
            p.requires_grad = False

        print("Starting Training...")
        for i in range(step_done, self.p.niters):
            #self.tracker.epoch_start()
            for p in self.netD.parameters():
                p.requires_grad = True
            for _ in range(self.p.iterD):    
                data = next(gen)
                real = data.to(self.device).unsqueeze(dim=1)
                self.netD.zero_grad()
                with autocast():
                    if self.p.hinge:
                        noise = torch.randn(real.shape[0], self.p.z_size, 1, 1,1,
                                    dtype=torch.float, device=self.device)

                        if self.p.stylegan:
                            fake, _ = self.netG(noise)
                            real.requires_grad_()
                        else:
                            fake = self.netG(noise)

                        real_out = self.netD(real)

                        errD_real = (nn.ReLU()(1.0 - real_out)).mean()
                        errD_fake = (nn.ReLU()(1.0 + self.netD(fake))).mean()
                        errD = errD_fake + errD_real

                        if self.p.stylegan:
                            gradients = torch.autograd.grad(outputs=real_out, inputs=real,
                                                   grad_outputs=torch.ones(real_out.size(), device=real.device),
                                                   create_graph=False, retain_graph=True)[0]

                            gradients = gradients.reshape(real.shape[0], -1)
                            gp = 10 * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

                            errD = errD + gp
                    else:
                        noise = torch.randn(real.shape[0], self.p.z_size, 1, 1,1,
                                        dtype=torch.float, device=self.device)
                        fake = self.netG(noise)
                        errD_real = self.netD(real).mean()
                        errD_fake = self.netD(fake).mean()
                        gradient_penalty = self.calc_gradient_penalty(real.data, fake.data)
                        errD =  errD_fake - errD_real + gradient_penalty

                self.scalerD.scale(errD).backward()
                self.scalerD.step(self.optimizerD)
                self.scalerD.update()

            for p in self.netD.parameters():
                p.requires_grad = False

            for p in self.netG.parameters():
                p.requires_grad = True

            self.netG.zero_grad()
            with autocast():
                noise = torch.randn(real.shape[0], self.p.z_size, 1, 1,1,
                            dtype=torch.float, device=self.device)
                if self.p.stylegan:
                    ws = self.netG.module.mapping(noise)
                    fake = self.netG.module.synthesis(ws)
                    #fake, ws = self.netG(noise)
                else:
                    fake = self.netG(noise)

                errG = -self.netD(fake).mean()

                if self.p.stylegan:
                    num_pixels = fake.shape[2] * fake.shape[3] * fake.shape[4]
                    pl_noise = torch.randn(fake.shape, device=self.p.device) / np.sqrt(num_pixels)
                    outputs = (fake * pl_noise).sum()

                    pl_grads = torch.autograd.grad(outputs=outputs, inputs=ws,
                                          grad_outputs=torch.ones(outputs.shape, device=self.p.device),
                                          create_graph=False, retain_graph=True)[0]

                    pl = (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()
                    avg_pl_length = np.mean(pl.detach().cpu().numpy())

                    if self.pl_mean is not None:
                        pl_loss = ((pl - self.pl_mean) ** 2).mean()
                        if not torch.isnan(pl_loss):
                            errG = errG + pl_loss
                
            self.scalerG.scale(errG).backward()
            self.scalerG.step(self.optimizerG)
            self.scalerG.update()

            for p in self.netG.parameters():
                p.requires_grad = False
            #self.tracker.epoch_end()

            if self.p.stylegan:
                self.pl_mean = self.pl_length_ema.update_average(self.pl_mean, avg_pl_length)

            self.G_losses.append(errG.item())
            self.D_losses.append((errD_real.item(), errD_fake.item()))

            self.log(i, fake, real)
            if i%100 == 0 and i>0:
                self.fid_epoch.append(np.array(self.fid).mean())
                self.fid = []
                self.save_checkpoint(i)
        #self.tracker.stop()
        self.log_final(i, fake, real)
        print('...Done')
