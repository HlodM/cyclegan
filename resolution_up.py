import torch
import cv2
import config
import torchvision.transforms as tt
import numpy as np
from esrgan import RRDBNet


path = f"{config.dir_to_save}/RRDB_PSNR_x4.pth"
device = config.device
transform = tt.ToTensor()
stats = config.stats

net = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23)
net.load_state_dict(torch.load(path, map_location=device), strict=True)
net.eval()
net.to(device)


# read images
def sr_image(save_path):
    img = cv2.imread(save_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_lr = img.unsqueeze(0)
    img_lr = img_lr.to(device)

    with torch.no_grad():
        output = net(img_lr).data.squeeze().float().cpu().clamp_(0, 1).numpy()

    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(f"{config.dir_to_save}/images/sr_image.jpg", output)

    with open(f"{config.dir_to_save}/images/sr_image.jpg", 'rb') as p:
        res = p.read()

    return res
