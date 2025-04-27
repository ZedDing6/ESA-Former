import torch
import torch.nn as nn
from rich.console import Console

console=Console()
class SelfAttention(nn.Module):
    def __init__(self, in_dim,embed_dim):
        super(SelfAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.embedding = nn.Linear(1,embed_dim)
        self.sigmoid = nn.Sigmoid()

        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(in_dim,1)


    def forward(self, x):
        x1 = self.avg_pool(x)
        x2 = self.max_pool(x)
        Xq = self.embedding(x1.squeeze(dim=-1))
        Xk = self.embedding(x2.squeeze(dim=-1))
        q = self.query(Xq)
        k = self.query(Xk)

        attention_weights = torch.matmul(q, k.permute(0, 2, 1))
        attention_weights = attention_weights / q.size(-1) ** 0.5
        transposed_attention = torch.transpose(attention_weights, 1, 2)
        v =self.value(transposed_attention)
        v = self.sigmoid(v)
        v = v.view(v.size(0), v.size(1), 1, 1)
        attended = x*v
        return attended


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.attention =SelfAttention(5,32)
        self.classifier = nn.Sequential(
            nn.Linear(18432, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224(8,5,200,200)
        x=self.attention(x)
        x = self.features(x)
        # N x 512 x 7 x 7
        x4 = torch.flatten(x, start_dim=1)
        # print(x.size())
        # N x 512*7*7
        x = self.classifier(x4)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 5
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model
