import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torchvision
from PIL import Image

from Network import EONSS


def get_patches(image, output_size, stride):
    w, h = image.size[:2]
    new_h, new_w = output_size, output_size
    stride_h, stride_w = stride, stride

    h_start = np.arange(0, h - new_h, stride_h)
    w_start = np.arange(0, w - new_w, stride_w)

    patches = [image.crop((wv_s, hv_s, wv_s + new_w, hv_s + new_h)) for hv_s in h_start for wv_s in w_start]

    to_tensor = torchvision.transforms.ToTensor()
    patches = [to_tensor(patch) for patch in patches]
    patches = torch.stack(patches, dim=0)
    return patches


class Tester(object):
    def __init__(self, config):
        self.config = config
        self.use_cuda = torch.cuda.is_available() and self.config.use_cuda

        # initialize the model
        self.model = EONSS()

        if self.use_cuda:
            print("[*] Using GPU")
            self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
        else:
            print("[*] Using CPU")
            self.model.cpu()
        print("[*] Network initialized")

        # load the model
        self._load_checkpoint(ckpt="EONSS_model.pt")
        print("[*] Model loaded")

    def eval(self):

        if os.path.isfile(self.config.img):
            image = Image.open(self.config.img)
        else:
            raise Exception("[!] no image found at '{}'".format(self.config.img))

        t1 = time.time()

        image_patches = get_patches(image, 235, 128)

        image_patches = torch.autograd.Variable(image_patches)
        if self.use_cuda:
            image_patches = image_patches.cuda()

        score_predict = self.model(image_patches).cpu().data
        score_predict = torch.squeeze(score_predict, dim=1).numpy()
        score_predict_mean = np.mean(score_predict)

        print("[-] Image name:\t\t", self.config.img)
        print("[-] EONSS score:\t", score_predict_mean)
        print("[-] Time consumed:\t %.4f s" % (time.time() - t1))

        if self.config.save_result:
            with open("EONSS_result.txt", 'w') as txt_file:
                txt_file.write("Image name:\t\t" + str(self.config.img) + "\n" + "EONSS score:\t" + str(score_predict_mean))

    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            if not torch.cuda.is_available():
                state_dict = torch.load(ckpt, map_location='cpu')['state_dict']
            else:
                state_dict = torch.load(ckpt)['state_dict']
            if not self.use_cuda:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.` in the state_dict which is saved with the "nn.DataParallel()"
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(state_dict)
        else:
            raise Exception("[!] no checkpoint found at '{}'".format(ckpt))
