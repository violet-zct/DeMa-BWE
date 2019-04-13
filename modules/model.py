import torch.nn as nn
import torch
from tools.utils import init_linear_layer

class Discriminator(nn.Module):

    def __init__(self, args, lang=None):
        super(Discriminator, self).__init__()

        self.lang = lang # For further extension to multilingual embedding case
        self.emb_dim = args.emb_dim
        self.dis_layers = args.dis_layers
        self.dis_hid_dim = args.dis_hid_dim
        self.dis_dropout = args.dis_dropout
        self.dis_input_dropout = args.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            init_linear_layer(args, layers[-1])
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)
    
    
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.s2t_save_to = None
        self.t2s_save_to = None

    def set_save_s2t_path(self, path):
        self.s2t_save_to = path

    def set_save_t2s_path(self, path):
        self.t2s_save_to = path