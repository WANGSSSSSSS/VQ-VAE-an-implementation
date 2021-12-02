import torch
import torch.nn as nn
import argparse

from vqvae import model_zoo
from tqdm import tqdm
from torch.optim import SGD
from torch.utils.data import DataLoader

def train(model : nn.Module, dataset, lr, save_path):
    data = DataLoader(dataset, shuffle=True, num_workers=2, pin_memory=True)
    optimizer = SGD(model.parameters(), lr=lr)
    for i in range(100):
        with tqdm(data, unit="batch") as batches :
            for batch in batches:
                batches.set_description(f"epoch {i}")
                batch = batch.cuda()
                optimizer.zero_grad()
                _, loss = model(batch)
                loss.backward()
                optimizer.step()
                batches.set_postfix(loss=loss.item())

def build(args):
    encoder = 1
    decoder = 1
    vnum = args.vnum
    vdim = args.vdim
    model = model_zoo[args.vqe](encoder, encoder, vnum, vdim)

    save_path = f"{args.encoder}-{args.decoder}-{args.vae}-({args.dataset}).pth"

    return model, dataset, save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser("vq-vae")
    parser.add_argument("encoder", type=str)
    parser.add_argument("decoder", type=str)
    parser.add_argument("vae", type=str)
    parser.add_argument("dataset", type=str)
    parser.add_argument("--vnum", type=int, default=2560)
    parser.add_argument("--vdim", type=int, default=128)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--save", type=str, default="save")

    args = parser.parse_args()
    model, dataset, save_path = build(args)
    train(model, dataset, args.lr, save_path)




