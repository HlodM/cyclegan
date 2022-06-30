import os
from torchvision.utils import save_image
import torchvision.transforms as tt
import torch
from model import Generator
import config
from PIL import Image
import matplotlib.pyplot as plt

transform = config.transform_test
stats = config.stats
dir_to_save = config.dir_to_save
device = config.device
gen_paint = Generator().to(device)
tr = tt.Resize(config.image_size*2)


def denorm(img_tensor):
    return img_tensor * stats[1][0] + stats[0][0]


def change_image(image, path=f"{dir_to_save}/gen_photo.pth"):
    gen_paint.load_state_dict(torch.load(path, map_location=device))
    photo = denorm(gen_paint(transform(image))).detach().cpu()
    if not os.path.exists(f"{dir_to_save}/images/"):
        os.mkdir(f"{dir_to_save}/images")
    save_path = f"{dir_to_save}/images/image.jpg"
    save_image(photo, save_path)
    with open(f"{dir_to_save}/images/image.jpg", 'rb') as changed:
        res = changed.read()
    return res, save_path


# ima = Image.open('e:/download/ima001.jpg')
# photo, _ = change_image(ima)
# print(photo)
# # plt.imshow(photo.permute(1, 2, 0))
# # plt.show()
