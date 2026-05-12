from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
os.environ['CUDA_DEVICE_ORDER'] = '0,1'
import torch
import random
# my_devs = '0,1'
# os.environ['CUDA_VISIBLE_DEVICES'] = my_devs
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import json
import torch.utils.data
from torchvision.transforms import transforms as T
from lib.opts import opts
from lib.models.model import create_model, load_model, save_model
from lib.models.data_parallel import DataParallel
from lib.logger import Logger
from lib.datasets.dataset_factory import get_dataset
from lib.trains.train_factory import train_factory

def run(opt):
    torch.manual_seed(opt.seed)
    # np.random.seed(opt.seed)
    # random.seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

    print('Setting up data...')
    Dataset = get_dataset(opt.dataset, opt.task)  # if opt.task==mot -> JointDataset

    f = open(opt.data_cfg)  # choose which dataset to train '../src/lib/cfg/mot15.json',
    data_config = json.load(f)
    trainset_paths = data_config['train']  # 训练集路径
    dataset_root = data_config['root']  # 数据集所在目录
    print("Dataset root: %s" % dataset_root)
    f.close()

    # Image data transformations
    use_imagenet_norm = 'lwdetr' in opt.arch
    if use_imagenet_norm:
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transforms = T.Compose([T.ToTensor()])

    # Dataset
    dataset = Dataset(opt=opt,
                      root=dataset_root,
                      paths=trainset_paths,
                      img_size=opt.input_wh,
                      augment=True,
                      transforms=transforms)
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print("opt:\n", opt)
    logger = Logger(opt)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str  # 多GPU训练
    print("opt.gpus_str: ", opt.gpus_str)
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')  # 设置GPU


    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)

    # Split optimizer: backbone gets smaller LR with AdamW, heads get full LR
    _use_split_opt = 'lwdetr' in opt.arch and hasattr(model, 'backbone')
    if _use_split_opt:
        backbone_param_ids = {id(p) for p in model.backbone.parameters()}
        backbone_params = [p for p in model.parameters() if id(p) in backbone_param_ids]
        other_params = [p for p in model.parameters() if id(p) not in backbone_param_ids]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': opt.lr * opt.backbone_lr_scale, 'weight_decay': 0.05},
            {'params': other_params, 'lr': opt.lr, 'weight_decay': 0.0},
        ])
        print(f'Using split AdamW optimizer: backbone lr={opt.lr * opt.backbone_lr_scale:.2e}, '
              f'heads lr={opt.lr:.2e}')

    # Load LW-DETR COCO pretrained backbone weights if provided
    if opt.backbone_weights and hasattr(model, 'backbone'):
        from lib.models.model import load_lwdetr_backbone
        model = load_lwdetr_backbone(model, opt.backbone_weights)
    else:
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    start_epoch = 0
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model(model,
                                                   opt.load_model,
                                                   optimizer,
                                                   opt.resume,
                                                   opt.lr,
                                                   opt.lr_step)

    # Get dataloader
    if opt.is_debug:

        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   drop_last=True)  # debug时不设置线程数(即默认为0)
    else:

        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   drop_last=True)  # debug时不设置线程数(即默认为0)

    print('Starting training...')
    Trainer = train_factory[opt.task]
    trainer = Trainer(opt=opt, model=model, optimizer=optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'

        # Backbone freeze/unfreeze schedule
        if _use_split_opt:
            if epoch <= opt.freeze_backbone_epochs:
                for p in model.backbone.parameters():
                    p.requires_grad_(False)
                if epoch == start_epoch + 1 or epoch == 1:
                    print(f'Backbone frozen for epochs 1-{opt.freeze_backbone_epochs}')
            elif epoch == opt.freeze_backbone_epochs + 1:
                for p in model.backbone.parameters():
                    p.requires_grad_(True)
                print(f'Backbone unfrozen at epoch {epoch}')

        # Train an epoch
        log_dict_train, _ = trainer.train(epoch, train_loader)

        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))

        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last' + opt.arch + '.pth'),
                           epoch, model, optimizer)
        logger.write('\n')

        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

            decay = 0.1 ** (opt.lr_step.index(epoch) + 1)
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * decay
            print('Drop LR by factor', decay)

        if epoch % 5 == 0 or epoch >= 25:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    print("opt.gpus: ", opt.gpus)
    print('epoch:', opt.num_epochs)
    run(opt)
