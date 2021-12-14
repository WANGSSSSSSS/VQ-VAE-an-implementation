import os
import random

import numpy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import argparse

from vqvae import model_zoo as vqvae_zoo
from encoder import model_zoo as encoder_zoo
from decoder import model_zoo as decoder_zoo

from tqdm import tqdm
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

def train(model : nn.Module, dataset, lr:float, save_path:str) -> None:
    model = model.cuda()
    data = DataLoader(dataset,batch_size=100, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = SGD(model.parameters(), lr=lr)
    for i in range(1, 2000):
        with tqdm(data, unit="batch") as batches :
            for batch, _ in batches:
                batches.set_description(f"epoch {i}")
                batch = batch.cuda()
                optimizer.zero_grad()
                _, loss = model(batch)
                loss.backward()
                optimizer.step()
                batches.set_postfix(loss=loss.item())
        if i%5 == 0 :
            torch.save( model.state_dict(), save_path)

def visulize(model: nn.Module,dataset : Dataset, save_path : str,):
    import matplotlib.pyplot as plt
    from torchvision.transforms import ToTensor, ToPILImage
    cp = torch.load(save_path, map_location="cpu")
    model.load_state_dict(cp)
    model.eval()
    #fig, ax = plt.subplots(10,2)
    plt.subplots_adjust(wspace=0,hspace=0)
    for i in range(12):
        index = random.randint(0, len(dataset))
        image, _ = dataset[index]
        out, _ = model(image[None, ...])

        out_image = ToPILImage()(out[0])
        image = ToPILImage()(image)

        plt.subplot(4, 6, i*2 + 1)
        plt.imshow(out_image)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(4, 6, i*2 + 2)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
    plt.show()


def build(args):
    vnum = args.vnum
    vdim = args.vdim
    encoder = encoder_zoo[args.encoder](2, 1, 64, 64)
    decoder = decoder_zoo[args.decoder](2, 64, 64, 1)
    model = vqvae_zoo[args.vae](encoder, decoder, vnum, vdim)

    from torchvision.datasets import CIFAR100, MNIST
    from torchvision.transforms import PILToTensor,Compose, ConvertImageDtype
    transform = Compose(
        [
            PILToTensor(),
            #lambda x : x.float()
            ConvertImageDtype(torch.float)
        ]
    )
    dataset = MNIST("./data", True, download=True, transform=transform)
    #dataset = CIFAR100("./data", True, download=True, transform=transform)
    save_path = args.save + f"/{args.encoder}-{args.decoder}-{args.vae}-({args.dataset}).pth"
    return model, dataset, save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("vq-vae")
    parser.add_argument("--encoder", type=str, default="Simple")
    parser.add_argument("--decoder", type=str, default="Simple")
    parser.add_argument("--vae", type=str, default="disVQ")
    parser.add_argument("--dataset", type=str, default="cifar100")
    parser.add_argument("--vnum", type=int, default=2)
    parser.add_argument("--vdim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--save", type=str, default="save")

    args = parser.parse_args()
    model, dataset, save_path = build(args)

    visulize(model, dataset, save_path)
    #train(model, dataset, args.lr, save_path)




