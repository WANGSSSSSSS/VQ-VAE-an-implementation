import torch
import torch.nn as nn
import torch.nn.functional as F


class disVQ(nn.Module):
    def __init__(self, encoder, decoder, num=2560, dim=128):
        super(disVQ, self).__init__()
        self.encoder:nn.Module = encoder
        self.decoder:nn.Module = decoder
        self.vq = nn.Embedding(num, dim)
        
        self.num = num
        self.dim = dim
        self.vq.weight.data.uniform_(-1.0 / num, 1.0 / num)

    def sample(self, one_hot):
        return one_hot

    def forward(self, batch):
        z:torch.Tensor = self.encoder(batch)

        b,c,h,w = z.shape
        z_flatten = torch.permute(z, [0, 2, 3, 1])
        z_flatten = z_flatten.contiguous()
        z_flatten = z_flatten.view(-1, self.dim)
        dis = torch.sum(z_flatten**2, dim=1, keepdim=True) + \
              torch.sum(self.vq.weight**2, dim=1) - \
              2* z_flatten @ self.vq.weight.t()
        ind = torch.argmin(dis, dim=1)
        one_hot = F.one_hot(ind, num_classes=self.num)
        if not self.training :
            one_hot = self.sample(one_hot)
        one_hot = one_hot.to(torch.float)
        z_q = one_hot @ self.vq.weight
        z_q = z_q.view(b,h,w,c)
        z_q = z_q.permute([0,3,1,2]).contiguous()

        vq_out = z + (z_q - z).detach()
        out = self.decoder(vq_out)
        loss = None
        if self.training :
            loss = 0.0
            loss += torch.mean((z_q - z.detach())**2)
            loss += torch.mean((z_q.detach() - z)**2)
            loss += torch.mean((batch - out)**2)

        return  out, loss
