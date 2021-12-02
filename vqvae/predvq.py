import torch
import torch.nn as nn
import torch.nn.functional as F


class prdVQ(nn.Module):
    def __init__(self, encoder, decoder, num=2560, dim=128):
        super(prdVQ, self).__init__()
        self.encoder: nn.Module = encoder
        self.decoder: nn.Module = decoder

        self.predict: nn.Module = nn.Sequential(
            nn.Linear(dim,num),
            nn.Softmax(),
        )

        self.vq = nn.Embedding(num, dim)

        self.num = num
        self.dim = dim
        self.embedding.weight.data.uniform_(-1.0 / num, 1.0 / num)

    def sample(self, one_hot):

        return one_hot

    def forward(self, batch):
        z: torch.Tensor = self.encoder(batch)
        b, c, h, w = z.shape
        z_flatten = torch.permute(z, [0, 2, 3, 1])
        z_flatten = z_flatten.contiguous()
        z_flatten = z_flatten.view(-1, self.dim)

        logits = self.predict(z_flatten)

        ind = torch.argmin(logits, dim=1)
        one_hot = F.one_hot(ind, num_classes=ind)

        if not self.training:
            one_hot = self.sample(one_hot)

        z_q = one_hot @ self.vq.weight
        z_q = z_q.view(b, h, w, c)
        z_q = z_q.permute([0, 3, 1, 2]).contiguous()

        vq_out = z + (z_q - z).deatch()
        out = self.decoder(vq_out)

        loss = None
        if self.training:
            loss = 0.0
            loss += torch.mean((z_q - z.detach()) ** 2)
            loss += torch.mean((z_q.detach() - z) ** 2)
            loss += torch.mean((batch - out) ** 2)

        return out, loss