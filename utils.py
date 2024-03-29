import os
import random
import time
from os import listdir
from os.path import join

import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image

from imageio import imread, imsave


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


INIT_TIMES = 100
LAT_TIMES = 1000


def measure_latency_in_ms(model, input_shape, is_cuda):
    lat = AverageMeter()
    model.eval()

    x = torch.randn(input_shape)
    if is_cuda:
        model = model.cuda()
        x = x.cuda()
    else:
        model = model.cpu()
        x = x.cpu()

    with torch.no_grad():
        for _ in range(INIT_TIMES):
            output = model(x)

        for _ in range(LAT_TIMES):
            tic = time.time()
            output = model(x)
            toc = time.time()
            lat.update(toc - tic, x.size(0))

    return lat.avg * 1000  # save as ms


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    mod = num_imgs % BATCH_SIZE

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = imread(path, pilmode=mode)
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')

    if height is not None and width is not None:
        image = np.array(Image.fromarray(image).resize((height, width)))
    return image


pil_to_tensor = transforms.ToTensor()


def get_batch(paths):
    img_ir = []
    img_vis = []
    w_list = []
    for i in range(len(paths)):
        # ir
        pil_ir = Image.open(paths[i][0]).convert('L')
        tensor_ir = pil_to_tensor(pil_ir).unsqueeze(0)
        img_ir.append(tensor_ir)
        # vis
        pil_vis = Image.open(paths[i][1]).convert('L')
        tensor_vis = pil_to_tensor(pil_vis).unsqueeze(0)
        img_vis.append(tensor_vis)
        # vsm
        wlist = np.load(paths[i][2])
        w_list.append(wlist)
    ir_batch = torch.cat(img_ir, dim=0)
    vis_batch = torch.cat(img_vis, dim=0)
    return ir_batch, vis_batch, w_list

def get_train_images_auto(paths):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = Image.open(path)
        mode = image.mode
        if mode == 'RGB':
            image = image.convert('L')
        image = np.reshape(image, [1, image.size[1], image.size[0]])

        images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()

    return images




def get_test_images(paths, height=None, width=None, mode='L'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = ImageToTensor(image)
        else:
            image = ImageToTensor(image)
        images.append(image)
    images = torch.stack(images, dim=0)
    return images


def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')

    torch.save(state, filename)

    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')

        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        print('x.size:', x.shape)
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
