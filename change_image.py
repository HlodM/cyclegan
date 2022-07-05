import os
import numpy as np
import torch
from torchvision.utils import save_image
from PIL import Image
import cv2
from model import Generator
from esrgan import RRDBNet
import train_params


stats = train_params.stats
device = train_params.device
dir_to_save = train_params.dir_to_save
vg_path = train_params.vg_path

gen_paint = Generator().to(device)
net = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)
net.load_state_dict(torch.load(train_params.esrgan_path, map_location=device), strict=True)
net.eval()
net.to(device)

if not os.path.exists(f"{dir_to_save}/images/"):
    os.mkdir(f"{dir_to_save}/images")


def denorm(img_tensor):
    return img_tensor * stats[1][0] + stats[0][0]


def change_image(image, gen_root=train_params.gen_path, save_path=vg_path):
    gen_paint.load_state_dict(torch.load(gen_root, map_location=device))
    photo = denorm(gen_paint(train_params.transform_test(image))).detach().cpu()
    save_image(photo, save_path)


def sr_image(in_path, out_path=train_params.sr_path):
    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_lr = img.unsqueeze(0)
    img_lr = img_lr.to(device)

    with torch.no_grad():
        output = net(img_lr).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(out_path, output)


if __name__ == '__main__':
    test_image = Image.open(train_params.image_path)
    change_image(test_image)
    out = Image.open(vg_path)
    if train_params.sr:
        sr_image(vg_path)
        out = Image.open(train_params.sr_path)
    out.show()
