import os.path
import torch
import torch.nn as nn
import torchvision.transforms as tt

batch_size = 1
image_size = 128
res_blocks_nums = 6   # 6 for image_size=128 or 9 for image_size=256+
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
lr = 0.0002
epochs = 1

lambda_cycle = 10
lambda_ident = 5

paint_dir = 'h:/My Drive/vangogh2photo/vangogh2photo/trainA/'
photo_dir = 'h:/My Drive/vangogh2photo/vangogh2photo/trainB/'

l1 = nn.L1Loss()
mse = nn.MSELoss()

# dir to save models weights
dir_to_save = f"{os.path.dirname(__file__)}/weights/"
if not os.path.exists(dir_to_save):
    os.mkdir(dir_to_save)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

transform = tt.Compose([tt.Resize(image_size), tt.RandomHorizontalFlip(), tt.ToTensor(), tt.Normalize(*stats)])
transform_test = tt.Compose([tt.Resize(image_size), tt.ToTensor(), tt.Normalize(*stats)])
load_model = True

TOKEN = '5423427869:AAHnsvTZK4KHN0fMgKCAnEt06RKzlT2sBxM'
