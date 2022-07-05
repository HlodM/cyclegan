import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
import train_params
from model import Generator, Discriminator, weights_init
from dataset import ImageDataset


def train(train_dl, gen_photo, gen_paint, disc_photo, disc_paint, optim_gen, optim_disc_photos, optim_disc_paints, l1,
          mse, epochs, decay, lr, lambda_cycle, lambda_ident, buffer_size, device, dir_to_save, checkpoints):

    # Losses
    losses_gen = []
    losses_disc_photo = []
    losses_disc_paint = []

    buffer_fake_photos = []
    buffer_fake_paints = []

    lr_step = lr/(epochs-decay)

    for epoch in range(epochs):
        loss_d_per_epoch_photo = []
        loss_d_per_epoch_paint = []
        loss_g_per_epoch = []

        for photos, paints in tqdm(train_dl):
            torch.cuda.empty_cache()

            photos = photos.to(device)
            paints = paints.to(device)

            # Train Generator
            optim_gen.zero_grad()

            identity_paints = gen_paint(paints)
            loss_ident_paints = l1(identity_paints, paints)

            identity_photos = gen_photo(photos)
            loss_ident_photos = l1(identity_photos, photos)

            fake_paints = gen_paint(photos)
            fake_paints_d = disc_paint(fake_paints)
            loss_disc_paints = mse(fake_paints_d, torch.ones_like(fake_paints_d))

            fake_photos = gen_photo(paints)
            fake_photos_d = disc_photo(fake_photos)
            loss_disc_photos = mse(fake_photos_d, torch.ones_like(fake_photos_d))

            reconstr_photos = gen_photo(fake_paints)
            cycle_loss_photos = l1(reconstr_photos, photos)

            reconstr_paints = gen_paint(fake_photos)
            cycle_loss_paints = l1(reconstr_paints, paints)

            loss_gen = (loss_disc_paints + cycle_loss_photos * lambda_cycle + loss_ident_paints * lambda_ident +
                        loss_disc_photos + cycle_loss_paints * lambda_cycle + loss_ident_photos * lambda_ident)
            loss_gen.backward()
            optim_gen.step()
            loss_g_per_epoch.append(loss_gen.item())

            # Train Discriminator paints
            optim_disc_paints.zero_grad()

            paints_d = disc_paint(paints)
            loss_paints_d = mse(paints_d, torch.ones_like(paints_d))

            fake_paints = gen_paint(photos)
            fake_paints_d = disc_paint(fake_paints)
            if len(buffer_fake_paints) < buffer_size:
                buffer_fake_paints.extend([torch.unsqueeze(el, 0) for el in fake_paints.detach().clone()])
                loss_fake_paints_d = mse(fake_paints_d.detach(), torch.zeros_like(fake_paints_d))
            else:
                idxs = random.sample(range(len(buffer_fake_paints)), len(paints))
                fake_paints = torch.cat([buffer_fake_paints[i] for i in idxs])
                fake_paints_d = disc_paint(fake_paints.detach())
                loss_fake_paints_d = mse(fake_paints_d.detach(), torch.zeros_like(fake_paints_d))
                buffer_fake_paints[:len(paints)] = [torch.unsqueeze(el, 0) for el in fake_paints.detach().clone()]

            loss_disc_paints = (loss_paints_d + loss_fake_paints_d) / 2
            loss_disc_paints.backward()
            optim_disc_paints.step()
            loss_d_per_epoch_paint.append(loss_disc_paints.item())

            # Train Discriminator photos
            optim_disc_photos.zero_grad()

            photos_d = disc_photo(photos)
            loss_photos_d = mse(photos_d, torch.ones_like(photos_d))

            fake_photos = gen_photo(paints)
            fake_photos_d = disc_photo(fake_photos)
            if len(buffer_fake_photos) < buffer_size:
                buffer_fake_photos.extend([torch.unsqueeze(el, 0) for el in fake_photos])
                loss_fake_photos_d = mse(fake_photos_d, torch.zeros_like(fake_photos_d))
            else:
                idxs = random.sample(range(len(buffer_fake_photos)), len(photos))
                fake_photos = torch.cat([buffer_fake_photos[i].clone() for i in idxs])
                fake_photos_d = disc_photo(fake_photos.detach())
                loss_fake_photos_d = mse(fake_photos_d, torch.zeros_like(fake_photos_d))
                buffer_fake_photos[:len(photos)] = [torch.unsqueeze(el, 0) for el in fake_photos]

            loss_disc_photos = (loss_photos_d + loss_fake_photos_d) / 2
            loss_disc_photos.backward()
            optim_disc_photos.step()
            loss_d_per_epoch_photo.append(loss_disc_photos.item())

        # Record losses
        losses_gen.append(np.mean(loss_g_per_epoch))
        losses_disc_photo.append(np.mean(loss_d_per_epoch_photo))
        losses_disc_paint.append(np.mean(loss_d_per_epoch_paint))

        if checkpoints:
            if not os.path.exists(f"{dir_to_save}/checkpoints/"):
                os.mkdir(f"{dir_to_save}/checkpoints")
            torch.save(gen_photo.state_dict(), f"{dir_to_save}/checkpoints/gen_photo_epoch_{epoch}.pth")
            torch.save(gen_paint.state_dict(), f"{dir_to_save}/checkpoints/gen_paint_epoch_{epoch}.pth")
            torch.save(disc_photo.state_dict(), f"{dir_to_save}/checkpoints/disc_photo_epoch_{epoch}.pth")
            torch.save(disc_paint.state_dict(), f"{dir_to_save}/checkpoints/disc_paint_epoch_{epoch}.pth")

        if epoch+1 > decay:
            lr -= lr_step

        # Log losses
        print("Epoch [{}/{}], loss_gen: {:.4f}, loss_disc_photo: {:.4f}, loss_disc_paint: {:.4f}".
              format(epoch + 1, epochs, losses_gen[-1], losses_disc_photo[-1], losses_disc_paint[-1]))

    torch.save(gen_photo.state_dict(), f"{dir_to_save}/gen_photo.pth")
    torch.save(gen_paint.state_dict(), f"{dir_to_save}/gen_paint.pth")
    torch.save(disc_photo.state_dict(), f"{dir_to_save}/disc_photo.pth")
    torch.save(disc_paint.state_dict(), f"{dir_to_save}/disc_paint.pth")

    return losses_gen, losses_disc_photo, losses_disc_paint


def seed_everything(seed=train_params.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    seed_everything()
    lr = train_params.lr
    dir_to_save = train_params.dir_to_save
    device = train_params.device
    print(device)

    gen_photo = Generator().to(device)
    gen_paint = Generator().to(device)
    disc_photo = Discriminator().to(device)
    disc_paint = Discriminator().to(device)

    l1 = train_params.l1.to(device)
    mse = train_params.mse.to(device)

    if train_params.load_model:
        gen_photo.load_state_dict(torch.load(f"{dir_to_save}/gen_photo.pth", map_location=device))
        gen_paint.load_state_dict(torch.load(f"{dir_to_save}/gen_paint.pth", map_location=device))
        disc_photo.load_state_dict(torch.load(f"{dir_to_save}/disc_photo.pth", map_location=device))
        disc_paint.load_state_dict(torch.load(f"{dir_to_save}/disc_paint.pth", map_location=device))
    else:
        gen_photo.apply(weights_init)
        gen_paint.apply(weights_init)
        disc_photo.apply(weights_init)
        disc_paint.apply(weights_init)

    optim_gen = torch.optim.Adam(list(gen_photo.parameters()) + list(gen_paint.parameters()), lr=lr, betas=(0.5, 0.999))
    optim_disc_photos = torch.optim.Adam(disc_photo.parameters(), lr=lr, betas=(0.5, 0.999))
    optim_disc_paints = torch.optim.Adam(disc_paint.parameters(), lr=lr, betas=(0.5, 0.999))

    dataset = ImageDataset(train_params.photo_dir, train_params.paint_dir, transform=train_params.transform)
    train_dl = DataLoader(dataset, train_params.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    history = train(train_dl, gen_photo, gen_paint, disc_photo, disc_paint, optim_gen, optim_disc_photos,
                    optim_disc_paints, l1, mse, train_params.epochs, train_params.decay, lr, train_params.lambda_cycle,
                    train_params.lambda_ident, train_params.buffer_size, device, dir_to_save, train_params.checkpoints)


if __name__ == '__main__':
    main()
