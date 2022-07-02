import os.path
import torch
import torch.nn as nn
import torchvision.transforms as tt


# to get TOKEN type '/start' message to 'https://t.me/BotFather'
# for more information visit 'https://core.telegram.org/bots'
TOKEN = '5423427869:AAHnsvTZK4KHN0fMgKCAnEt06RKzlT2sBxM'

# dirs with training data
paint_dir = 'h:/My Drive/vangogh2photo/vangogh2photo/trainA/'
photo_dir = 'h:/My Drive/vangogh2photo/vangogh2photo/trainB/'

# dir to save models parameters
dir_to_save = f"{os.path.dirname(__file__)}/weights/"
if not os.path.exists(dir_to_save):
    os.mkdir(dir_to_save)

batch_size = 1
image_size = 128
res_blocks_num = 6   # 6 for image_size=128 or 9 for image_size=256+
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
lr = 0.0002
epochs = 4
decay = 2
buffer_size = 50

lambda_cycle = 10
lambda_ident = 0

seed = 42

l1 = nn.L1Loss()
mse = nn.MSELoss()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

transform = tt.Compose([tt.Resize(image_size), tt.RandomHorizontalFlip(), tt.ToTensor(), tt.Normalize(*stats)])
transform_test = tt.Compose([tt.Resize(image_size), tt.ToTensor(), tt.Normalize(*stats)])

load_model = False
checkpoints = False
sr = True  # applying esrgan to increase resolution

# path to image if you need to test the model directly from IDE by change_image function without using the telegram bot
image_path = 'e:/download/ima007.jpg'
