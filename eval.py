import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import src
from src.paths import CHECKPOINTS_DIR, DATA_DIR
from src.dataloader import get_dataloader
from src.resnet_wider import resnet50x1, resnet50x2, resnet50x4
from src.lifecycles import get_device, save_stats, load_stats, save_model, load_model
from src.viz_helper import compare_training_stats, save_plt

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Finetuning of SimCLR Checkpoints')
parser.add_argument('-a', '--arch', default='resnet50-1x')
parser.add_argument('-e', '--epochs', default=20, type=int)
parser.add_argument('--exp-name', default='default')
parser.add_argument('-d', '--dataset', choices=('CIFAR-10', 'CIFAR-100', 'STL-10'), default='CIFAR-10')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

best_acc1 = 0


def main():
    args = parser.parse_args()

    # create model
    if args.arch == 'resnet50-1x':
        model = resnet50x1()
        sd = 'resnet50-1x.pth'
    elif args.arch == 'resnet50-2x':
        model = resnet50x2()
        sd = 'resnet50-2x.pth'
    elif args.arch == 'resnet50-4x':
        model = resnet50x4()
        sd = 'resnet50-4x.pth'
    else:
        raise NotImplementedError
    sd = torch.load(sd, map_location='cpu')
    model.load_state_dict(sd['state_dict'])

    # update linear head to have correct number of output classes
    num_classes = int(args.dataset.split('-')[-1])
    model.fc = nn.Linear(model.fc.weight.shape[-1], num_classes)


    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).to('cuda')
        cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()


    # Data loading code
    train_loader, val_loader = get_dataloader(args.dataset, args.batch_size, DATA_DIR)

    train_top1_metrics = {}
    train_top5_metrics = {}
    val_top1_metrics = {}
    val_top5_metrics = {}
    for i in range(args.epochs):
        print(f'---- Epoch {i + 1} Training ----')
        train_top1_epoch_metrics = {}
        train_top5_epoch_metrics = {}
        val_top1_epoch_metrics = {}
        val_top5_epoch_metrics = {}
        top1, top5 = train(train_loader, model, criterion, args)
        train_top1_epoch_metrics['acc'] = top1.item()
        train_top5_epoch_metrics['acc'] = top5.item()
        print(f'---- Epoch {i + 1} Validation ----')
        top1, top5 = validate(val_loader, model, criterion, args)
        val_top1_epoch_metrics['acc'] = top1.item()
        val_top5_epoch_metrics['acc'] = top5.item()

        # update aggregate metrics
        train_top1_metrics[f'epoch_{i + 1}'] = train_top1_epoch_metrics
        train_top5_metrics[f'epoch_{i + 1}'] = train_top5_epoch_metrics
        val_top1_metrics[f'epoch_{i + 1}'] = val_top1_epoch_metrics
        val_top5_metrics[f'epoch_{i + 1}'] = val_top5_epoch_metrics

    save_path = f'{args.dataset}_{args.exp_name}_train_top1_metrics'
    save_stats(train_top1_metrics, save_path)
    save_path = f'{args.dataset}_{args.exp_name}_train_top5_metrics'
    save_stats(train_top5_metrics, save_path)

    save_path = f'{args.dataset}_{args.exp_name}_val_top1_metrics'
    save_stats(val_top1_metrics, save_path)
    save_path = f'{args.dataset}_{args.exp_name}_val_top5_metrics'
    save_stats(val_top5_metrics, save_path)

    # Models have run, lets plot the stats
    all_stats = []
    labels = []
    load_path = f'{args.dataset}_{args.exp_name}_train_top1_metrics'
    all_stats.append(load_stats(load_path))
    labels.append('top1_acc')
    load_path = f'{args.dataset}_{args.exp_name}_train_top5_metrics'
    all_stats.append(load_stats(load_path))
    labels.append('top5_acc')

    # for metric in ('top1_acc', 'top5_acc'):
        # For every config, plot the loss across number of epochs
    metric = 'acc'
    plt = compare_training_stats(all_stats, labels, metric_to_compare=metric, y_label=metric, title=f'{args.dataset} Accuracy vs Epoch (Train)')
    save_plt(plt, f'{args.dataset}_{args.exp_name}_{metric}_train')
    plt.clf()

    # plt validation stats
    all_stats = []
    labels = []
    load_path = f'{args.dataset}_{args.exp_name}_val_top1_metrics'
    all_stats.append(load_stats(load_path))
    labels.append('top1_acc')
    load_path = f'{args.dataset}_{args.exp_name}_val_top5_metrics'
    all_stats.append(load_stats(load_path))
    labels.append('top5_acc')

    # for metric in ('top1_acc', 'top5_acc'):
        # For every config, plot the loss across number of epochs
    metric = 'acc'
    plt = compare_training_stats(all_stats, labels, metric_to_compare=metric, y_label=metric, title=f'{args.dataset} Accuracy vs Epoch (Validation)')
    save_plt(plt, f'{args.dataset}_{args.exp_name}_{metric}_val')
    plt.clf()

    # aggregate final epoch test accuracies across experiments and save
    test_acc = {}
    load_path = f'{args.dataset}_{args.exp_name}_val_metrics'
    stats = load_stats(load_path)
    exp = f'{args.dataset}_{args.exp_name}'
    # for metric in ('top1_acc', 'top5_acc'):
        # test_acc[f'{exp}_{metric}'] = stats[f'epoch_{args.epochs}'][metric]
    # save_stats(test_acc, f'{args.dataset}_{args.exp_name}_test_acc')

def train(train_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix='Train: ')

    # switch to evaluate mode
    device = get_device()
    model.train()
    optim = torch.optim.Adam(model.parameters())

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        optim.zero_grad()
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        optim.step()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    device = get_device()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
