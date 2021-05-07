import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class CnnRnn(nn.Module):
    def __init__(self, cnn, rnn, seq_per_img, use_fc):
        super(CnnRnn, self).__init__()
        self.cnn = cnn
        self.rnn = rnn
        self.seq_per_img = seq_per_img
        self.use_fc = use_fc

    def forward(self, img, labels):
        att_feats = self.cnn(img).permute(0, 2, 3, 1)
        fc_feats = att_feats.mean(2).mean(1)

        att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0), self.seq_per_img,) \
                                            + att_feats.size()[1:])).contiguous().view(
                                            *((att_feats.size(0) * self.seq_per_img,) + att_feats.size()[1:]))
        if self.use_fc == 0:
            fc_feats = Variable(torch.FloatTensor(1, 1, 1, 1).cuda())
        else :
            fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), self.seq_per_img,) \
                                                  + fc_feats.size()[1:])).contiguous().view(
                                                    *((fc_feats.size(0) * self.seq_per_img,) + fc_feats.size()[1:]))
        output = self.rnn(fc_feats, att_feats, labels)
        return output

