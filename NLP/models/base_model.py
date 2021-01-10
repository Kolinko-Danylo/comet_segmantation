from torch import nn
import torchvision
import torch

class VGGExtractor(nn.Module):
    def __init__(self, embedding_size, pretraned=True):
        super(VGGExtractor, self).__init__()

        base_model = torchvision.models.vgg11_bn(pretrained=pretraned)
        self.features = base_model.features

        self.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.embedding = nn.Linear(in_features=512, out_features=embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[:-2])
        x = self.fc(x)
        x = self.embedding(x)
        return x / torch.norm(x, dim=-1, keepdim=True)


class ExtractorRNN(nn.Module):

    def __init__(self, input_size, output_size, n_hidden=650, n_layers=3,
                 drop_prob=0.4):
        super(ExtractorRNN, self).__init__()
        self.input_size = input_size
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.output_size = output_size

        self.rnn = nn.GRU(self.input_size, n_hidden, n_layers,
                          dropout=drop_prob, batch_first=True)

        self.vgg = VGGExtractor(128)

    def forward(self, x, hidden):
        r_output, hidden = self.rnn(x, hidden)

        out = r_output[:, -self.output_size:].unsqueeze(1)
        rgb_batch = out.repeat(1, 3, 1, 1)
        fout = self.vgg(rgb_batch)
        return fout