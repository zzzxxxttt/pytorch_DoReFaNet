import os
import time
from datetime import datetime
# from termcolor import colored
# import colorama

import torch


def load_pretrain(net, optimizer, pretrain_dir):
  pretrain_file = os.path.join(pretrain_dir, 'checkpoint.t7')
  if os.path.isfile(pretrain_file):
    pretrained_state = torch.load(pretrain_file)
    net.load_state_dict(pretrained_state['state_dict'])
    if optimizer is not None:
      optimizer.load_state_dict(pretrained_state['optimizer'])
    print("=> loaded checkpoint '%s'" % pretrain_file)
  else:
    print("=> no checkpoint found at '%s'" % pretrain_file)


def save_model(net, optimizer, checkpoint_dir, name='checkpoint'):
  torch.save({'state_dict': net.state_dict(),
              'optimizer': optimizer.state_dict(), },
             os.path.join(checkpoint_dir, '%s.t7' % name))
  print('model saved in %s' % checkpoint_dir)


def print_train_log(epoch, step, losses, exp_per_sec, batch_per_sec):
  format_str = '%s epoch: %d step: %d ' % (str(datetime.now())[:-7], epoch, step)
  for key in losses.keys():
    format_str += '%s= %.5f ' % (key, losses[key])
  format_str += ' (%d samples/sec %.2f sec/batch)' % (exp_per_sec, batch_per_sec)
  print(format_str)


def print_eval_log(precision_1, precision_5=None):
  if precision_5 is not None:
    print('\n%s----------------------------------------------------------------------------'
          ' Precision@1: %.2f%% Precision@5: %.2f%% \n' %
          (str(datetime.now())[:-7], precision_1, precision_5))
  else:
    print('\n%s----------------------------------------------------------------------------'
          ' Precision@1: %.2f%%\n' % (str(datetime.now())[:-7], precision_1))


