import torch
import torch.nn as nn
import argparse

from vqvae import model_zoo as vqvae_zoo
from encoder import model_zoo as encoder_zoo
from decoder import model_zoo as decoder_zoo

from tqdm import tqdm
from torch.optim import SGD
from torch.utils.data import DataLoader

def train(model : nn.Module, dataset, lr:float, save_path:str) -> None:
    model = model.cuda()
    data = DataLoader(dataset,batch_size=100, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = SGD(model.parameters(), lr=lr)
    for i in range(1, 100):
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

def build(args):
    vnum = args.vnum
    vdim = args.vdim
    encoder = encoder_zoo[args.encoder](3, 3, 64, 64)
    decoder = decoder_zoo[args.decoder](3, 64, 64, 3)
    model = vqvae_zoo[args.vae](encoder, decoder, vnum, vdim)

    from torchvision.datasets import CIFAR100
    from torchvision.transforms import ToTensor,Compose
    transform = Compose(
        [
            ToTensor(),
        ]
    )
    dataset = CIFAR100("./data", True, download=True, transform=transform)
    save_path = args.save + f"/{args.encoder}-{args.decoder}-{args.vae}-({args.dataset}).pth"
    return model, dataset, save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("vq-vae")
    parser.add_argument("--encoder", type=str, default="Simple")
    parser.add_argument("--decoder", type=str, default="Simple")
    parser.add_argument("--vae", type=str, default="disVQ")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--vnum", type=int, default=2560)
    parser.add_argument("--vdim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save", type=str, default="save")

    args = parser.parse_args()
    model, dataset, save_path = build(args)
    train(model, dataset, args.lr, save_path)




