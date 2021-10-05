import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

import torch.optim
import torchvision
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import Dataset

from dataset import Food_LT
from model.model import resnet34, FocalLoss
from model.densenet import densenet121,densenet161
from model.SWTDN import swtdn2
import config as cfg
from utils import adjust_learning_rate, save_checkpoint, train, validate, logger
from torch.cuda import amp

def main():
    model = densenet161()
    
    if cfg.resume:
        state_dict = torch.load(cfg.root+'/ckpt/current.pth.tar')
        pretrained_dict = state_dict['state_dict_model']
        model_dict = model.state_dict()
        state_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict.keys()}
        # 更新权重
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('resume from current.pth.tar')

    print('log save at:' + cfg.log_path)
    logger('', init=True)
    logger(str(model))
    if not torch.cuda.is_available():
        logger('Plz train on cuda !')
        os._exit(0)



    print('Load dataset ...')
    dataset = Food_LT(False, root=cfg.root, batch_size=cfg.batch_size, num_works=4)

    train_loader = dataset.train_instance
    val_loader = dataset.eval
    
    criterion = FocalLoss()
    optimizer = torch.optim.SGD([{"params": model.parameters()}], cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)

    if cfg.gpu is not None:
        print('Use cuda !')
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
        criterion = criterion.cuda(cfg.gpu)

    scaler = amp.GradScaler()

    best_acc = 0
    for epoch in range(cfg.num_epochs):
        logger('--'*10 + f'epoch: {epoch}' + '--'*10)
        logger('Training start ...')
        
        adjust_learning_rate(optimizer, epoch, cfg)
        
        train(train_loader, model, criterion, optimizer, epoch, scaler)
        logger('Wait for validation ...')
        acc = validate(val_loader, model, criterion)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        logger('* Best Prec@1: %.3f%%' % (best_acc))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict_model': model.state_dict(),
            'best_acc': best_acc,
        }, is_best, cfg.root)

    print('Finish !')


if __name__ == '__main__':
    main()