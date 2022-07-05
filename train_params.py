import os.path
import torch
import torch.nn as nn
import torchvision.transforms as tt


# to get TOKEN type '/start' message to 'https://t.me/BotFather'
# for more information visit 'https://core.telegram.org/bots'
TOKEN =   # something like '1111227869:AAHaadVFK4HHN0fMgKCAnEt06RKztT7sNxM'

# dirs with training data
paint_dir =
photo_dir =

# path to image if you need to test the model directly from IDE by change_image function without using the telegram bot
image_path =

batch_size = 1
image_size = 128
res_blocks_num = 6   # 6 for image_size=128 or 9 for image_size=256+
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
lr = 0.0002
epochs = 200
decay = 100
buffer_size = 50

lambda_cycle = 10
lambda_ident = 1

seed = 42

l1 = nn.L1Loss()
mse = nn.MSELoss()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

transform = tt.Compose([tt.Resize(image_size), tt.RandomHorizontalFlip(), tt.ToTensor(), tt.Normalize(*stats)])
transform_test = tt.Compose([tt.Resize(image_size), tt.ToTensor(), tt.Normalize(*stats)])

load_model = True
checkpoints = False
sr = False  # applying esrgan to increase resolution

# dir to save models parameters
dir_to_save = f"{os.path.dirname(__file__)}/weights/"
if not os.path.exists(dir_to_save):
    os.mkdir(dir_to_save)

# path to generator and esrgan
gen_path = f"{dir_to_save}/gen_photo.pth"
esrgan_path = f"{dir_to_save}/RRDB_PSNR_x4.pth"

# paths to save vangogh styled image, super resolution and initial
vg_path = f"{dir_to_save}/images/vg_image.jpg"
sr_path = f"{dir_to_save}/images/sr_image.jpg"
bot_path = f"{dir_to_save}/images/bot_image.jpg"
