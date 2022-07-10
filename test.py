import torch
from model import FM
from os.path import join
from os import listdir
import PIL.Image as Image
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
from os.path import exists
from utils import get_train_images_auto, get_test_images
import cv2
import genotypes
import utils
import time
tensor_to_pil = transforms.ToPILImage()


model_dir = r'.\checkpoint'
model_path = join(model_dir, 'weights_epoch_50.pt')

genotype_en1 = eval('genotypes.%s' % 'genotype1')
genotype_en2 = eval('genotypes.%s' % 'genotype2')
genotype_de = eval('genotypes.%s' % 'genotype3')

model = FM(16, genotype_en1, genotype_en2, genotype_de).cuda()

params = torch.load(model_path)

model.load_state_dict(params)
model.eval()

image_dir = r'images'
save_dir = r'.\Result'

if not exists(save_dir):
    os.mkdir(save_dir)


with torch.no_grad():
    total = 0
    for i in range(21):
        image_ir_path = join(image_dir, 'ir', str(i+1)+'.png')
        image_vis_path = join(image_dir, 'vis', str(i+1)+'.png')

        tensor_ir = get_test_images(image_ir_path).cuda()
        tensor_vis = get_test_images(image_vis_path).cuda()

        t1 = time.time()
        tensor_f = model(tensor_ir, tensor_vis)
        t2 = time.time()

        image_tensor = tensor_f.cpu().squeeze()
        image_tensor = torch.clamp(image_tensor, 0, 1)
        print(i)
        image_pil = tensor_to_pil(image_tensor)
        image_pil.save(join(save_dir, 'TNO_'+str(i+1)+'.jpg'))
        total += t2 - t1


    print("param size = %fMB", utils.count_parameters_in_MB(model))
    print('Total Time cost: {}'.format(total))
    print('Average time cost per image pair:{}'.format(total/21))
