import glob
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, dir_photos, dir_paints, transform=None):
        self.transform = transform
        self.photos = glob.glob(dir_photos + '/*.*')
        self.paints = glob.glob(dir_paints + '/*.*')

    def __getitem__(self, index):
        if self.transform is not None:
            photo = self.transform(Image.open(self.photos[index % len(self.photos)]))
            paint = self.transform(Image.open(self.paints[index % len(self.paints)]))
        else:
            photo = Image.open(self.photos[index % len(self.photos)])
            paint = Image.open(self.paints[index % len(self.paints)])

        return photo, paint

    def __len__(self):
        return 10
        # return max(len(self.photos), len(self.paints))
