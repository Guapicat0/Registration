import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import cv2
from  CycleGAN.models import Generator
from CycleGAN.datasets import ImageDataset
import numpy
from PIL import Image
def cyclegan(input_img):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/RGB2INF/', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=512, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', default='cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_A2B', type=str, default='CycleGAN/output/netG_A2B.pth',
                        help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth',
                        help='B2A generator checkpoint file')
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc)


    if opt.cuda:
        netG_A2B.cuda()


    # Load state dicts
    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))


    # Set model's test mode
    netG_A2B.eval()


    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor

    ###################################
    img = cv2.imread(input_img)
    img = cv2.resize(img,(512,512))
    trans = transforms.ToTensor()
    img = trans(img)
    img = img.unsqueeze(dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img.to(device)
    fake_B = 0.5 * (netG_A2B(img).data + 1.0)


    return fake_B



    ###################################

if __name__ =="__main__":
    fake_B=cyclegan(input_img="F:/Registering/CycleGAN/datasets/RGB2INF/train/A/3.jpg")
    print(fake_B.shape)
    outputRs = fake_B.permute(0, 2, 3, 1)

    mm = outputRs.cpu().detach().numpy()*255
    mm = mm.astype("uint8").reshape(512,512,3)
    print(mm.shape)
    cv2.imshow("Matches", mm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()





