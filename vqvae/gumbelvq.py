import torch
import torch.nn as nn
import torch.nn.functional as F


class gumbelVQ(nn.Module):
    def __init__(self, encoder, decoder, num=2560, dim=128, ):
        super(gumbelVQ, self).__init__()
        self.encoder: nn.Module = encoder
        self.decoder: nn.Module = decoder

        self.predict: nn.Module = nn.Sequential(
            nn.Linear(dim,num),
            nn.Softmax(),
        )
        self.max_loss = 0.11
        self.last_loss = 0.001

        self.vq = nn.Embedding(num, dim)

        self.num = num
        self.dim = dim
        self.vq.weight.data.uniform_(-1.0 / num, 1.0 / num)

    def sample(self, ind, thre=0.1):
        cond  = torch.rand_like(ind) < thre
        out = torch.where(cond, 0.9, ind)
        return out

    def forward(self, batch):
        z: torch.Tensor = self.encoder(batch)
        b, c, h, w = z.shape
        z_flatten = torch.permute(z, [0, 2, 3, 1])
        z_flatten = z_flatten.contiguous()
        z_flatten = z_flatten.view(-1, self.dim)

        distr = torch.sum(z_flatten, 1) /(b*h*w)
        logits = self.predict(z_flatten)

        if not self.training:
            logits = self.sample(logits)
        gumbel_logits = F.gumbel_softmax(logits, tau=self.last_loss /self.max_loss, hard=True)



        vq_out = gumbel_logits @ self.vq.weight
        out = self.decoder(vq_out)

        loss = None
        if self.training:
            loss = 0.0
            loss += distr*torch.log(distr * self.num + 0.001)
            loss += torch.mean((batch - out) ** 2)
            self.last_loss = loss.item()
            self.max_loss = self.last_loss if self.max_loss < self.max_loss else self.max_loss


        return out, loss