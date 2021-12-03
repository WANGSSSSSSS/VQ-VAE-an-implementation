import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, in_c):
        super(Residual, self).__init__()
        self.layers = [
            nn.Conv2d(in_c, in_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, in_c, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(in_c),
        ]

        self.layers = nn.Sequential(*self.layers)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.layers(x))


class SimplexN(nn.Module):
    def __init__(self, n, in_c, hidden_c, out_c):
        super(SimplexN, self).__init__()
        layers = []

        layers.extend(
            [
                nn.Conv2d(in_c, hidden_c, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(hidden_c),
            ]
        )

        for i in range(4):
            layers.append(Residual(hidden_c))

        for i in range(n):
            layers.extend([
                nn.Upsample(scale_factor=2),
                nn.Conv2d(hidden_c, hidden_c, kernel_size=(5, 5), stride=(1, 1), padding=2),
                nn.BatchNorm2d(hidden_c),
                nn.ReLU()
                # nn.ConvTranspose2d(hidden_c, hidden_c, kernel_size=(2, 2), stride=(2, 2)),
                # nn.BatchNorm2d(hidden_c),
                # nn.ReLU(),
            ])

        layers.extend(
            [
                nn.Conv2d(hidden_c, out_c, kernel_size=(1, 1), stride=(1, 1)),
                #nn.Conv2d(hidden_c, out_c, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(out_c),
                nn.Sigmoid(),
            ]
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)