import os
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

import torch.optim
import torchvision
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset

from model import resnet34
import config as cfg
import time
import math


def process_bar(precent, clks, strs='', width=30):
    use_num = math.ceil(precent*width)
    space_num = int(width-use_num)
    precent = precent*100
    print(strs+' [%s%s]%3.0f%%     %3.2fs'%(use_num*'■', space_num*' ',precent,clks),file=sys.stdout,flush=True, end='\r')


transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class Testset(Dataset):
    num_classes = cfg.num_classes

    def __init__(self):
        self.img_path = []
        self.name = []
        self.transform = transform_test
        base = "data/food/test/"
        files= os.listdir(os.path.join(cfg.root,base)) #得到文件夹下的所有文件名称
        for file in files: #遍历文件夹
            self.img_path.append(os.path.join(os.path.join(cfg.root,base),file))
            self.name.append(file)
        print("load %d images in test folder"%(len(self.img_path)))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        path = self.img_path[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample.unsqueeze(0),self.name[index]

def main():
    model = resnet34()
    state_dict = torch.load(cfg.root+'/ckpt/model_best.pth.tar')
    model.load_state_dict(state_dict['state_dict_model'])

    if not torch.cuda.is_available():
        logger('Plz train on cuda !')
        os._exit(0)

    if cfg.gpu is not None:
        print('Use cuda !')
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)

    dataset = Testset()
    # switch to evaluate mode
    model.eval()

    class_num = torch.zeros(cfg.num_classes).cuda()
    pred_class = np.array([])
    filename = 'result.txt'
    start = time.time()
    with torch.no_grad() and open (filename, 'w') as file_object:
        file_object.write("Id, Expected\n")
        for i, (images,name) in enumerate(dataset):
            images = images.cuda(cfg.gpu, non_blocking=True)
            output = model(images)
            _, predicted = output.max(1)
            file_object.write(name+", "+str(predicted.cpu().numpy()[0])+"\n")
            process_bar(i/len(dataset), time.time()-start, name)
    
    print("\nFinish !  time:"+time.time()-start+"s")


if __name__ == '__main__':
    main()