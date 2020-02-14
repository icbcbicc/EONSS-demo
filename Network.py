import math

import torch.nn as nn
import torch.nn.functional as F

from GDN import Gdn


class EONSS(nn.Module):
    def __init__(self):
        super(EONSS, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.gdn1 = Gdn(8)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.gdn2 = Gdn(16)
        self.conv3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.gdn3 = Gdn(32)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.gdn4 = Gdn(64)

        self.st2_fc1 = nn.Conv2d(64, 256, 1, stride=1, padding=0)
        self.st2_gdn1 = Gdn(256)
        self.st2_fc2 = nn.Conv2d(256, 1, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Gdn):
                m.gamma.data.fill_(1)
                m.beta.data.fill_(1e-2)

    def forward(self, x):
        batch_size = x.size()[0]

        x = F.max_pool2d(self.gdn1(self.conv1(x)), (2, 2))
        x = F.max_pool2d(self.gdn2(self.conv2(x)), (2, 2))
        x = F.max_pool2d(self.gdn3(self.conv3(x)), (2, 2))
        x = F.max_pool2d(self.gdn4(self.conv4(x)), (2, 2))

        y2 = self.st2_gdn1(self.st2_fc1(x))
        s = self.st2_fc2(y2)
        s = s.view(batch_size, -1)

        return s
