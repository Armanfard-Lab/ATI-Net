import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
import torch.utils.data.sampler as sampler

from utils import *

class Decoder(nn.Module):
    ''' 
    Modular SegNet decoder used for task-specific heads in PAD-Net implimentation.
    '''
    def __init__(self, filter):
        super(Decoder, self).__init__()
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # define convolution layer
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                            self.conv_layer([filter[i], filter[i]])))
                
    def conv_layer(self, channel, pred=False):
            if not pred:
                conv_block = nn.Sequential(
                    nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_features=channel[1]),
                    nn.ReLU(inplace=True),
                )
            else:
                conv_block = nn.Sequential(
                    nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                    nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
                )
            return conv_block

    def forward(self, indices, g_maxpool):
        decoder, upsampl = ([0] * 5 for _ in range(2))
        for i in range(5):
            decoder[-i - 1] = [0] * 2

        for i in range(5):
                if i == 0:
                    upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                    decoder[i][0] = self.decoder_block[-i - 1](upsampl[i])
                    decoder[i][1] = self.conv_block_dec[-i - 1](decoder[i][0])
                else:
                    upsampl[i] = self.up_sampling(decoder[i - 1][-1], indices[-i - 1])
                    decoder[i][0] = self.decoder_block[-i - 1](upsampl[i])
                    decoder[i][1] = self.conv_block_dec[-i - 1](decoder[i][0])

        return decoder[-1][-1]

class PAD_Net(nn.Module):
    '''
    Our implimentation of a SegNet based PAD-Net.
    '''
    def __init__(self, tasks=['segmentation', 'depth', 'normal'], num_out_channels={'segmentation': 13, 'depth': 1, 'normal': 3}):
        super(PAD_Net, self).__init__()
        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.tasks = tasks
        self.num_out_channels = num_out_channels
        self.class_nb = num_out_channels['segmentation']

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                
        self.decoders = nn.ModuleList([Decoder(filter) for _ in self.tasks])

        # define distillation modules
        self.multi_modal_distillation = MultiTaskDistillationModule(self.tasks, self.tasks, filter[0])

        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block


    def forward(self, x):
        g_encoder, g_maxpool, indices = ([0] * 5 for _ in range(3))
        for i in range(5):
            g_encoder[i] = [0] * 2

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
        
        pred_features = {}
        for i in range(len(self.tasks)):
            pred_features['features_%s' %(self.tasks[i])] = self.decoders[i](indices, g_maxpool)

        # refine features through multi-modal distillation
        out = self.multi_modal_distillation(pred_features)

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(out[self.tasks[0]]), dim=1)
        t2_pred = self.pred_task2(out[self.tasks[1]])
        t3_pred = self.pred_task3(out[self.tasks[2]]) if 'normal' in self.tasks else 0
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True) if 'normal' in self.tasks else 0

        return [t1_pred, t2_pred, t3_pred]