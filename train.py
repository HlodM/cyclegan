import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import config
from model import Generator, Discriminator, weights_init, seed_everything
from dataset import ImageDataset


def train(train_dl, gen_photo, gen_paint, disc_photo, disc_paint, optim_gen_photo, optim_gen_paint, optim_disc_photos,
          optim_disc_paints, l1, mse, epochs, lr, lambda_cycle, lambda_ident, device, dir_to_save, checkpoints=False):

    # use decay 50 instead 100 from the original paper to speed up the program
    lr_step = lr/(epochs-50) if epochs > 50 else 0

    # Losses
    losses_gen_photo = []
    losses_gen_paint = []
    losses_disc_photo = []
    losses_disc_paint = []

    for epoch in range(epochs):
        loss_d_per_epoch_photo = []
        loss_d_per_epoch_paint = []
        loss_g_per_epoch_photo = []
        loss_g_per_epoch_paint = []

        for photos, paints in tqdm(train_dl):
            torch.cuda.empty_cache()

            photos = photos.to(device)
            paints = paints.to(device)

            # Train Generator paints
            optim_gen_paint.zero_grad()

            identity_paints = gen_paint(paints)
            loss_ident_paints = l1(identity_paints, paints)

            fake_paints = gen_paint(photos)
            fake_paints_d = disc_paint(fake_paints)
            loss_disc_paints = mse(fake_paints_d, torch.ones_like(fake_paints_d))

            reconstr_photos = gen_photo(fake_paints)
            cycle_loss_photos = l1(reconstr_photos, photos)

            loss_gen_paint = loss_disc_paints + cycle_loss_photos * lambda_cycle + loss_ident_paints * lambda_ident
            loss_gen_paint.backward()
            optim_gen_paint.step()
            loss_g_per_epoch_paint.append(loss_gen_paint.item())

            # train Discriminator paints
            optim_disc_paints.zero_grad()

            paints_d = disc_paint(paints)
            loss_paints_d = mse(paints_d, torch.ones_like(paints_d))

            fake_paints = gen_paint(photos)
            fake_paints_d = disc_paint(fake_paints)
            loss_fake_paints_d = mse(fake_paints_d, torch.zeros_like(fake_paints_d))

            loss_disc_paints = (loss_paints_d + loss_fake_paints_d) / 2
            loss_disc_paints.backward()
            optim_disc_paints.step()
            loss_d_per_epoch_paint.append(loss_disc_paints.item())

            # Train Generator photos
            optim_gen_photo.zero_grad()

            identity_photos = gen_photo(photos)
            loss_ident_photos = l1(identity_photos, photos)

            fake_photos = gen_photo(paints)
            fake_photos_d = disc_photo(fake_photos)
            loss_disc_photos = mse(fake_photos_d, torch.ones_like(fake_photos_d))

            reconstr_paints = gen_paint(fake_photos)
            cycle_loss_paints = l1(reconstr_paints, paints)

            loss_gen_photo = loss_disc_photos + cycle_loss_paints * lambda_cycle + loss_ident_photos * lambda_ident
            loss_gen_photo.backward()
            optim_gen_photo.step()
            loss_g_per_epoch_photo.append(loss_gen_photo.item())

            # train Discriminator photos
            optim_disc_photos.zero_grad()

            photos_d = disc_photo(photos)
            loss_photos_d = mse(photos_d, torch.ones_like(photos_d))

            fake_photos = gen_photo(paints)
            fake_photos_d = disc_photo(fake_photos)
            loss_fake_photos_d = mse(fake_photos_d, torch.zeros_like(fake_photos_d))

            loss_disc_photos = (loss_photos_d + loss_fake_photos_d) / 2
            loss_disc_photos.backward()
            optim_disc_photos.step()
            loss_d_per_epoch_photo.append(loss_disc_photos.item())

        # Record losses & scores
        losses_gen_paint.append(np.mean(loss_g_per_epoch_paint))
        losses_gen_photo.append(np.mean(loss_g_per_epoch_photo))
        losses_disc_photo.append(np.mean(loss_d_per_epoch_photo))
        losses_disc_paint.append(np.mean(loss_d_per_epoch_paint))

        if checkpoints:
            if not os.path.exists(f"{dir_to_save}/checkpoints/"):
                os.mkdir(f"{dir_to_save}/checkpoints")
            torch.save(gen_photo.state_dict(), f"{dir_to_save}/checkpoints/gen_photo_epoch_{epoch}.pth")
            torch.save(gen_paint.state_dict(), f"{dir_to_save}/checkpoints/gen_paint_epoch_{epoch}.pth")
            torch.save(disc_photo.state_dict(), f"{dir_to_save}/checkpoints/disc_photo_epoch_{epoch}.pth")
            torch.save(disc_paint.state_dict(), f"{dir_to_save}/checkpoints/disc_paint_epoch_{epoch}.pth")

        if epoch > 50:
            lr -= lr_step

        # Log losses
        print("""Epoch [{}/{}], loss_gen_paint: {:.4f}, loss_gen_photo: {:.4f}, loss_disc_photo: {:.4f},
              loss_disc_paint: {:.4f}""".format(epoch + 1, epochs, losses_gen_paint[-1], losses_gen_photo[-1],
                                                losses_disc_photo[-1], losses_disc_paint[-1]))

    torch.save(gen_photo.state_dict(), f"{dir_to_save}/gen_photo.pth")
    torch.save(gen_paint.state_dict(), f"{dir_to_save}/gen_paint.pth")
    torch.save(disc_photo.state_dict(), f"{dir_to_save}/disc_photo.pth")
    torch.save(disc_paint.state_dict(), f"{dir_to_save}/disc_paint.pth")

    return losses_gen_paint, losses_gen_photo, losses_disc_photo, losses_disc_paint


def main():
    seed_everything()
    batch_size = config.batch_size
    lr = config.lr
    epochs = config.epochs

    lambda_cycle = config.lambda_cycle
    lambda_ident = config.lambda_ident

    paint_dir = config.paint_dir
    photo_dir = config.photo_dir
    dir_to_save = config.dir_to_save

    l1 = config.l1
    mse = config.mse

    gen_photo = Generator()
    gen_paint = Generator()
    disc_photo = Discriminator()
    disc_paint = Discriminator()

    device = config.device
    print(device)

    gen_photo = gen_photo.to(device)
    gen_paint = gen_paint.to(device)
    disc_photo = disc_photo.to(device)
    disc_paint = disc_paint.to(device)
    l1 = l1.to(device)
    mse = mse.to(device)

    if config.load_model:
        gen_photo.load_state_dict(torch.load(f"{dir_to_save}/gen_photo.pth", map_location=device))
        gen_paint.load_state_dict(torch.load(f"{dir_to_save}/gen_paint.pth", map_location=device))
        disc_photo.load_state_dict(torch.load(f"{dir_to_save}/disc_photo.pth", map_location=device))
        disc_paint.load_state_dict(torch.load(f"{dir_to_save}/disc_paint.pth", map_location=device))
    else:
        gen_photo.apply(weights_init)
        gen_paint.apply(weights_init)
        disc_photo.apply(weights_init)
        disc_paint.apply(weights_init)

    optim_gen_photo = torch.optim.Adam(gen_photo.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_gen_paint = torch.optim.Adam(gen_paint.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_disc_photos = torch.optim.Adam(disc_photo.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_disc_paints = torch.optim.Adam(disc_paint.parameters(), lr=lr, betas=(0.5, 0.999))

    dataset = ImageDataset(paint_dir, photo_dir, transform=config.transform)
    train_dl = DataLoader(dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    history = train(train_dl, gen_photo, gen_paint, disc_photo, disc_paint, optim_gen_photo, optim_gen_paint,
                    optim_disc_photos, optim_disc_paints, l1, mse, epochs, lr, lambda_cycle, lambda_ident,
                    device, dir_to_save, checkpoints=True)


if __name__ == '__main__':
    main()
