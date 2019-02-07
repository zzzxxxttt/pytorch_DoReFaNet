import torchvision.transforms as transforms


def cifar_transform(is_training=True):
  if is_training:
    transform_list = [transforms.RandomHorizontalFlip(),
                      transforms.Pad(padding=4, padding_mode='reflect'),
                      transforms.ToPILImage(),
                      transforms.RandomCrop(32, padding=0),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
  else:
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]

  transform_list = transforms.Compose(transform_list)
  return transform_list


def imagenet_transform(is_training=True):
  if is_training:
    transform_list = transforms.Compose([transforms.RandomSizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  else:
    transform_list = transforms.Compose([transforms.Scale(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  return transform_list
