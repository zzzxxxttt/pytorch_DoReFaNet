# Pytorch implementation of DoReFa-Net

This repository is the pytorch implementation of [DoReFa-Net](https://arxiv.org/pdf/1606.06160.pdf) for neural network compression. 
The code is inspired by the original [tensorpack implementation](https://github.com/tensorpack/tensorpack/tree/master/examples/DoReFa-Net).
This implementation supports k-bit quantization for weights
(k-bit activation quantization is still under construction). 
 
## Requirements:
- python>=3.5
- pytorch>=0.4.1
- tensorboardX


## CIFAR-10:
(Quantized models are trained from scratch instead of finetune from pretrained model.)

Model|W-bit|A-bit|Accuracy
:---:|:---:|:---:|:---:
ResNet-20|32|32|92.13
ResNet-20|4|32|91.46
ResNet-20|2|32|91.05
ResNet-20|1|32|90.54

## ImageNet2012
(Quantized models are finetuned from pretrained model.)

Model|W-bit|A-bit|Top-1 Accuracy|Top-5 Accuracy
:---:|:---:|:---:|:---:|:---:
AlexNet|32|32|56.50%|79.01%
AlexNet|1|32|53.31%|76.72%


