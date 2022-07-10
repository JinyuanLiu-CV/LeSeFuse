import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import FM
import torch.nn.functional as F
from os.path import join
from os import listdir
import random
from PIL import Image
import pytorch_msssim
import math
import torchvision.transforms as transform
parser = argparse.ArgumentParser("Train")
import genotypes

parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='init learning rate')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=8, help='num of init channels')

parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

parser.add_argument('--dataset', type=str, default=r'.\TNO_RoadScene256', help='TNO')

args = parser.parse_args()
args.save = 'train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True

    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    mse_loss = torch.nn.MSELoss().cuda()
    ssim = pytorch_msssim.msssim

    genotype_en1 = eval('genotypes.%s' % 'genotype1')
    genotype_en2 = eval('genotypes.%s' % 'genotype2')
    genotype_de = eval('genotypes.%s' % 'genotype3')

    model = FM(16, genotype_en1, genotype_en2, genotype_de).cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.6)

    epochs = args.epochs

    Infrared_path_list1 = utils.list_images(os.path.join(args.dataset, 'TNOIr'))
    Visible_path_list1 = utils.list_images(os.path.join(args.dataset, 'TNOVis'))
    Infrared_path_list2 = utils.list_images(os.path.join(args.dataset, 'RoadSceneIr'))
    Visible_path_list2 = utils.list_images(os.path.join(args.dataset, 'RoadSceneVis'))

    Infrared_path_list = Infrared_path_list1 + Infrared_path_list2
    Visible_path_list = Visible_path_list1 + Visible_path_list2

    dir1 = os.path.join(args.dataset, 'TNOW')
    dir2 = os.path.join(args.dataset, 'RoadSceneW')

    vsm_list1 = [os.path.join(dir1, name) for name in listdir(dir1)]
    vsm_list2 = [os.path.join(dir2, name) for name in listdir(dir2)]
    vsm_list = vsm_list1 + vsm_list2

    queue = np.stack([Infrared_path_list, Visible_path_list, vsm_list], axis=1)

    random.shuffle(queue)
    train_queue, batches = utils.load_dataset(queue, args.batch_size)
    for epoch in range(epochs):

        # training
        train(train_queue, batches, args, model, ssim, mse_loss, optimizer, epoch)

        if (epoch+1) % 5 == 0:
            utils.save(model, os.path.join(args.save, 'weights_epoch_' + str(epoch+1) + '.pt'))


tensor_to_pil = transform.ToPILImage()
pil_to_tensor = transform.ToTensor()


def train(train_queue, batches, args, model, ssim, mse_loss, optimizer, epoch):
    for batch in range(batches):
        paths_train = train_queue[batch * args.batch_size:(batch * args.batch_size + args.batch_size)]  # 训练一批
        train_batch = utils.get_batch(paths_train)
        tensor_ir, tensor_vis, map_list = train_batch[0].cuda(), train_batch[1].cuda(), train_batch[2]
        outputs = model(tensor_ir, tensor_vis)

        optimizer.zero_grad()

        mseLoss = 0
        ssimLoss = 0

        for i in range(len(map_list)):
            map1 = torch.from_numpy(map_list[i][0]).unsqueeze(0).cuda()
            map2 = torch.from_numpy(map_list[i][1]).unsqueeze(0).cuda()

            input1 = tensor_ir[i]
            input2 = tensor_vis[i]
            output = outputs[i]
            vsm_img = input1 * map1 + input2 * map2
            mseLoss += mse_loss(vsm_img, output)
            ssimLoss += 1 - ssim(vsm_img.unsqueeze(0), output.unsqueeze(0), normalize=True, val_range=1)

        total_loss = mseLoss + 10*ssimLoss
        total_loss.backward()

        optimizer.step()
        logging.info("epoch: %d batch: %d total_loss: %f mse_loss: %f ssim_loss: %f ", epoch, batch, total_loss, mseLoss, ssimLoss)


if __name__ == '__main__':
    main()
