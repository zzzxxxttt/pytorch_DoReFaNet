import os
import time
import argparse
from datetime import datetime
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils.i_o import *
from nets.imgnet_alexnet import *
from imgnet_config import cfg

from tqdm import tqdm
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='DoReFa-Net pytorch')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='/mnt/tmp/raw-data')
parser.add_argument('--log_name', type=str, default='alexnet_all_w1o2_finetune')
parser.add_argument('--pretrain', action='store_true', default=True)
parser.add_argument('--pretrain_dir', type=str, default='./ckpt/alexnet_baseline')

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd', type=float, default=5e-4)

parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=100)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--use_gpu', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=20)

parser.add_argument('--cluster', action='store_true', default=False)

cfg = parser.parse_args()

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

if not cfg.cluster:
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
  os.environ["CUDA_VISIBLE_DEVICES"] = cfg.use_gpu


def main():
  # create model
  print("=> creating model...")
  model = AlexNet_Q().cuda()

  # define loss function (criterion) and optimizer
  optimizer = torch.optim.SGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [60, 80, 90, 95], gamma=0.1)
  criterion = nn.CrossEntropyLoss().cuda()

  # optionally resume from a checkpoint
  if cfg.pretrain:
    load_pretrain(model, optimizer, cfg.pretrain_dir)

  # Data loading code
  traindir = os.path.join(cfg.data_dir, 'train')
  valdir = os.path.join(cfg.data_dir, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

  train_dataset = datasets.ImageFolder(traindir, transforms.Compose([transforms.RandomResizedCrop(224),
                                                                     transforms.RandomHorizontalFlip(),
                                                                     transforms.ColorJitter(brightness=0.5,
                                                                                            contrast=0.5,
                                                                                            saturation=0.3),
                                                                     transforms.ToTensor(),
                                                                     normalize, ]))
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=cfg.train_batch_size,
                                             shuffle=True,
                                             num_workers=cfg.num_workers,
                                             pin_memory=True)

  val_dataset = datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(256),
                                                                 transforms.CenterCrop(224),
                                                                 transforms.ToTensor(),
                                                                 normalize]))
  val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=cfg.eval_batch_size,
                                           shuffle=False,
                                           num_workers=cfg.num_workers,
                                           pin_memory=True)

  summary_writer = SummaryWriter(cfg.log_dir)

  def train(epoch):
    # switch to train mode
    model.train()

    start_time = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
      # measure data loading time
      input_var = inputs.cuda()
      target_var = targets.cuda()

      # compute output
      output = model(input_var)
      loss = criterion(output, target_var)

      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if i % cfg.log_interval == 0:
        # measure elapsed time
        duration = time.time() - start_time
        start_time = time.time()
        num_per_sec = cfg.train_batch_size * cfg.log_interval / duration

        step = (epoch * len(train_loader) + i)
        print('%s: epoch: %d step: %d loss: %.3f (%d examples/sec)' % (datetime.now(), epoch, i, loss, num_per_sec))
        summary_writer.add_scalar('train_loss', loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)

  def validate(epoch):
    # switch to evaluate mode
    model.eval()
    top1 = 0
    top5 = 0

    with tqdm(total=len(val_dataset)) as pbar:
      for i, (inputs, targets) in enumerate(val_loader):
        targets = targets.cuda()
        input_var = inputs.cuda()

        # compute output
        output = model(input_var)

        # measure accuracy and record loss
        _, pred = output.data.topk(5, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        top1 += correct[:1].view(-1).float().sum(0, keepdim=True).item()
        top5 += correct[:5].view(-1).float().sum(0, keepdim=True).item()
        pbar.update(cfg.eval_batch_size)

    top1 *= 100 / len(val_dataset)
    top5 *= 100 / len(val_dataset)
    print('------------------------------------------------------------'
          '%s: Precision@1 %.3f%% Prec@5 %.3f%%' % (datetime.now(), top1, top5))
    summary_writer.add_scalar('Precision@1', top1, epoch)
    summary_writer.add_scalar('Precision@5', top5, epoch)

    return top1, top5

  for epoch in range(1, cfg.max_epochs):
    lr_schedu.step(epoch)
    train(epoch)
    validate(epoch)
    # remember best prec@1 and save checkpoint
    torch.save({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(), },
               os.path.join(cfg.ckpt_dir, 'checkpoint.t7'))


if __name__ == '__main__':
  main()
