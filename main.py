from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

print(torch.__version__)
print(torch.cuda.is_available())

# Set random seed for reproductibility
manualSeed = 999
print("Random seed:",manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = 'data/celeba/' #우리 각자의 환경에 맞는 경로를 적어주어야합니다!

workers = 2

batch_size = 64

image_size = 64

nc = 3

nz = 100

ngf = 64

ndf = 64

num_epochs = 10

lr = 0.0002

beta1 = 0.5

ngpu = 1

dataset = dset.ImageFolder(root=dataroot,
                           transform= transforms.Compose([
                              transforms.Resize(image_size),
                              transforms.CenterCrop(image_size),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

real_batch = next(iter(dataloader))
plt.figure(figsize=(10,10))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
#plt.show()

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator

class Generator(nn.Module):
  def __init__(self,ngpu):
    super(Generator, self).__init__()
    self.ngpu = ngpu
    # Deconvolution layer is ConvTranspose2d
    # output_size=stride * (input size-1)+filter size-2 * padding
    self.conv1=nn.Sequential(nn.ConvTranspose2d(in_channels=nz,out_channels=ngf*16,kernel_size=4,stride=1,padding=0, bias=False),
        nn.BatchNorm2d(ngf*16),
        nn.ReLU(True))
    self.conv2=nn.Sequential(nn.ConvTranspose2d(in_channels=ngf*16,out_channels=ngf*8,kernel_size=4,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(ngf*8),
        nn.ReLU(True))
    self.conv3=nn.Sequential(nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
        nn.BatchNorm2d(ngf*4),
        nn.ReLU(True))
    self.conv4=nn.Sequential(nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
        nn.BatchNorm2d(ngf*2),
        nn.ReLU(True))
    self.conv5=nn.Sequential( nn.ConvTranspose2d(ngf*2,nc,4,2,1,bias=False),
        nn.Tanh())
    """
    self.main = nn.Sequential(
        # input : z 벡터
        nn.ConvTranspose2d(in_channels=nz,out_channels=ngf*8,kernel_size=4,stride=1,padding=0, bias=False),
        nn.BatchNorm2d(ngf*8),
        nn.ReLU(True),
        # state size. (ngf*8)*4*4
        nn.ConvTranspose2d(ngf*8,ngf*4,4,2,1,bias=False),
        nn.BatchNorm2d(ngf*4),
        nn.ReLU(True),
        # (ngf*4)*8*8
        nn.ConvTranspose2d(ngf*4,ngf*2,4,2,1,bias=False),
        nn.BatchNorm2d(ngf*2),
        nn.ReLU(True),
        # (ngf*2)*16*16
        nn.ConvTranspose2d(ngf*2,ngf,4,2,1,bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),
        # ngf * 32 * 32
        nn.ConvTranspose2d(ngf,nc,4,2,1,bias=False),
        nn.Tanh()
        # nc * 64 *64
    )"""

  def forward(self,g_input):
      # print("g_input.size(): ",g_input.size()) # torch.Size([64, 100, 1, 1])
      g_input = self.conv1(g_input)
      # print("g_input.size(): ",g_input.size()) # torch.Size([64, 1024, 4, 4])
      g_input = self.conv2(g_input)
      # print("g_input.size(): ",g_input.size()) # torch.Size([64, 512, 8, 8])
      g_input = self.conv3(g_input)
      # print("g_input.size(): ",g_input.size()) # torch.Size([64, 256, 16, 16])
      g_input = self.conv4(g_input)
      # print("g_input.size(): ",g_input.size()) # torch.Size([64, 128, 32, 32])
      g_input = self.conv5(g_input)
      # print("g_input.size(): ",g_input.size()) # torch.Size([64, 3, 64, 64])
      return g_input

netG = Generator(ngpu).to(device)

if (device.type=='cuda') and (ngpu>1):
  netG = nn.DataParallel(netG, list(range(ngpu)))

netG.apply(weights_init)

print(netG)

# Discriminator

class Discriminator(nn.Module):
  def __init__(self,ngpu):
    super(Discriminator, self).__init__()
    self.ngpu = ngpu
    self.conv1 = nn.Sequential(
        # input : nc * 64 * 64
        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True))
    self.conv2 = nn.Sequential(# ndf * 32 * 32
        nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
        nn.BatchNorm2d(ndf*2),
        nn.LeakyReLU(0.2,inplace=True))
    self.conv3 = nn.Sequential(# (ndf*2) * 16 *16
        nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
        nn.BatchNorm2d(ndf*4),
        nn.LeakyReLU(0.2,inplace=True))
    self.conv4 = nn.Sequential(# (ndf*4)*8*8
        nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
        nn.BatchNorm2d(ndf*8),
        nn.LeakyReLU(0.2,inplace=True))
    self.conv5 = nn.Sequential(        # (ndf*8)*4*4
        nn.Conv2d(ndf*8,1,4,1,0,bias=False),
        nn.Sigmoid())
    """
    self.main = nn.Sequential(
        # input : nc * 64 * 64
        nn.Conv2d(nc,ndf,4,2,1,bias=False),
        nn.LeakyReLU(0.2,inplace=True),
        # ndf * 32 * 32
        nn.Conv2d(ndf,ndf*2,4,2,1,bias=False),
        nn.BatchNorm2d(ndf*2),
        nn.LeakyReLU(0.2,inplace=True),
        # (ndf*2) * 16 *16
        nn.Conv2d(ndf*2,ndf*4,4,2,1,bias=False),
        nn.BatchNorm2d(ndf*4),
        nn.LeakyReLU(0.2,inplace=True),
        # (ndf*4)*8*8
        nn.Conv2d(ndf*4,ndf*8,4,2,1,bias=False),
        nn.BatchNorm2d(ndf*8),
        nn.LeakyReLU(0.2,inplace=True),
        # (ndf*8)*4*4
        nn.Conv2d(ndf*8,1,4,1,0,bias=False),
        nn.Sigmoid()
    )
    """
  def forward(self,d_input):
      # print("d_input.size(): ",d_input.size()) # torch.Size([64, 3, 64, 64])
      d_input = self.conv1(d_input)
      # print("d_input.size(): ",d_input.size()) # torch.Size([64, 64, 32, 32])
      d_input = self.conv2(d_input)
      # print("d_input.size(): ",d_input.size()) # torch.Size([64, 128, 16, 16])
      d_input = self.conv3(d_input)
      # print("d_input.size(): ",d_input.size()) # torch.Size([64, 256, 8, 8])
      d_input = self.conv4(d_input)
      # print("d_input.size(): ",d_input.size()) # torch.Size([64, 512, 4, 4])
      d_input = self.conv5(d_input)
      # print("d_input.size(): ",d_input.size()) # torch.Size([64, 1, 1, 1])
      return d_input


netD = Discriminator(ngpu).to(device)

if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

netD.apply(weights_init)

print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(64,nz,1,1,device=device)

real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1,0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1,0.999))

# Training

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")

# for each epoch
for epoch in range(num_epochs):
    # for each batch
    for i, data in enumerate(dataloader, 0):

        # Update D

        # train with all-real batch
        netD.zero_grad()
        real_cpu = data[0].to(device)
        # print("real_cpu.size()",real_cpu.size()) # torch.Size([64, 3, 64, 64])
        b_size = real_cpu.size(0)
        # print("b_size", b_size) # 64
        label = torch.full((b_size,), real_label, device=device)

        output = netD(real_cpu)
        # print("output.size(): ", output.size()) # torch.Size([64, 1, 1, 1])
        output = output.view(-1)
        # print("output.size(): ",output.size()) # torch.Size([64])
        output=output.type(torch.FloatTensor).cuda()
        label = label.type(torch.FloatTensor).cuda()
        # print("label.size(): ", label.size()) # label.size():  torch.Size([64])
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with all-fake batch
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # print("noise.size(): ", noise.size()) # torch.Size([64, 100, 1, 1])
        fake = netG(noise)
        # print("fake.size(): ", fake.size()) # torch.Size([64, 3, 64, 64])
        label.fill_(fake_label)

        output = netD(fake.detach()).view(-1)
        # print("output.size(): ",output.size()) # torch.Size([64])
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        optimizerD.step()

        # Update G
        netG.zero_grad()
        label.fill_(real_label)

        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()

        optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # %%capture
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()


