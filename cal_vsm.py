# To generate the weight map in the dataset
import PIL.Image as Image
import numpy as np
import os

# where to generate
ir_dir = r'.\TNO_RoadScene64\Ir'
vis_dir = r'.\TNO_RoadScene64\Vis'
save_dir = r'.\TNO_RoadScene64\W'

name_list = os.listdir(ir_dir)


def calhis(img):
    ret = np.zeros(256, int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ret[img[i][j]] += 1
    return ret
def sal(his):
    ret = np.zeros(256, int)
    for i in range(256):
        for j in range(256):
            ret[i] += np.abs(j-i)*his[j]
    return ret

def vsm(img):
    his = np.zeros(256, int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            his[img[i][j]] += 1
    sal = np.zeros(256, int)
    for i in range(256):
        for j in range(256):
            sal[i] += np.abs(j-i)*his[j]
    map = np.zeros_like(img, int)
    for i in range(256):
        map[np.where(img == i)] = sal[i]
    if map.max()==0:
        return np.zeros_like(img, int)
    return map / (map.max())


def softmax(map1, map2, c):
    exp_x1 = np.exp(map1*c)
    exp_x2 = np.exp(map2*c)
    exp_sum = exp_x1 + exp_x2
    map1 = exp_x1/exp_sum
    map2 = exp_x2/exp_sum
    return map1, map2

for file_name in name_list:
    print(file_name)
    name = file_name.split('.')[0]
    if os.path.isfile(os.path.join(save_dir, name + '.npy')):
        continue
    ir_path = os.path.join(ir_dir, file_name)
    vis_path = os.path.join(vis_dir, file_name)
    img_ir = np.asarray(Image.open(ir_path).convert('L'))
    img_vis = np.asarray(Image.open(vis_path).convert('L'))
    map1 = vsm(img_ir)
    map2 = vsm(img_vis)
    # Linear transform
    # w1 = 0.5 + 0.5*(map1 - map2)
    # w2 = 0.5 + 0.5*(map2 - map1)

    # Softmax
    w1, w2 = softmax(map1, map2, c=5)

    w_list = np.stack([w1, w2], 0).astype(np.float16)
    # Save the weight map
    np.save(os.path.join(save_dir, name + '.npy'), w_list)

