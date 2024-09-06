import torch
from torch import nn, optim
from dataset import CustomImageDataset
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure

from model import Generator, Discriminator

def write_log_to_file(file_name, log):
    log_file = open(file_name, "w")
    log_file.write(str(log))
    log_file.close()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

epochs = 50
batch_size = 4
learning_rate = 0.0002
adam_betas = (0.9, 0.999)
l1_regularization_factor = 100

generator = Generator(12, 3, 128).to(device)
discriminator = Discriminator(12, 3, 128).to(device)
#generator.load_state_dict(torch.load("generator.pt"))
#discriminator.load_state_dict(torch.load("discriminator.pt"))

criterion = nn.BCELoss().to(device)
criterion_l1 = nn.L1Loss().to(device)

optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=adam_betas)
optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=adam_betas)
#optimizerG.load_state_dict(torch.load("optimG.pt"))
#optimizerD.load_state_dict(torch.load("optimD.pt"))

dataset_path = "sample_dataset/"
train_data = CustomImageDataset(dataset_path+"train/")
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = CustomImageDataset(dataset_path+"test/")
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

mse = MeanSquaredError().to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=(-1,1)).to(device)

real_label = torch.empty((4, 1, 24, 24), dtype=torch.float).fill_(1).to(device)
fake_label = torch.empty((4, 1, 24, 24), dtype=torch.float).fill_(0).to(device)

discriminator_loss_log = []
generator_loss_log = []

mse_log = []
ssim_log = []

last_save_meta_data_file = open("last_save_meta_data.txt", "w")

for i in range(0, epochs):
    for j, (image_depth, image_normal, image_diffuse, image_direct, image_global) in enumerate(train_dataloader):
        image_depth = image_depth.to(device)
        image_normal = image_normal.to(device)
        image_diffuse = image_diffuse.to(device)
        image_direct = image_direct.to(device)
        image_global = image_global.to(device)

        # Discriminator training
        discriminator.zero_grad()

        discriminator_output_real_image = discriminator(torch.cat((image_diffuse, image_direct, image_normal, image_depth, image_global), 1))
        discriminator_loss_real_image = criterion(discriminator_output_real_image, real_label)
        discriminator_loss_real_image.backward()

        fake_image = generator(torch.cat((image_diffuse, image_direct, image_normal, image_depth), 1))
        discriminator_output_fake_image = discriminator(torch.cat((image_diffuse, image_direct, image_normal, image_depth, fake_image.detach()),1))
        discriminator_loss_fake_image = criterion(discriminator_output_fake_image, fake_label)
        discriminator_loss_fake_image.backward()

        discriminator_loss = (discriminator_loss_real_image + discriminator_loss_fake_image) * 0.5 #logging
        optimizerD.step()

        # Generator training
        generator.zero_grad()

        discriminator_output_fake_image = discriminator(torch.cat((image_diffuse, image_direct, image_normal, image_depth, fake_image), 1))
        generator_loss = criterion(discriminator_output_fake_image, real_label) + l1_regularization_factor * criterion_l1(fake_image, image_global)
        generator_loss.backward()

        optimizerG.step()

        discriminator_loss_log.append(discriminator_loss.item())
        generator_loss_log.append(generator_loss.item())
        print("Done => epoch: " + str(i) + "/" + str(epochs) + "; batch: " + str(j) + "/" + str(len(train_dataloader)) + "; D_loss=" + str(discriminator_loss.item()) + "; G_loss=" + str(generator_loss.item()))

    # At the end of each epoch write loss logs to files, save model, and perform mse and ssim test
    print("Running tests...")
    with torch.no_grad():
        for z, (image_depth, image_normal, image_diffuse, image_direct, image_global) in enumerate(test_dataloader):
            image_depth = image_depth.to(device)
            image_normal = image_normal.to(device)
            image_diffuse = image_diffuse.to(device)
            image_direct = image_direct.to(device)
            image_global = image_global.to(device)

            input_test = torch.cat((image_diffuse, image_direct, image_normal, image_depth), 1)
            fake_image_test = generator(input_test)

            mse.update(fake_image_test, image_global)
            ssim.update(fake_image_test, image_global)

            print("Test batch: " + str(z) + "/" + str(len(test_dataloader)))

        mse_log.append(mse.compute().item())
        mse.reset()
        ssim_log.append(ssim.compute().item())
        ssim.reset()

    print("End of epoch " + str(i) + " MSE: " + str(mse_log[-1]))
    print("End of epoch " + str(i) + " SSIM: " + str(ssim_log[-1]))

    write_log_to_file("mse_log.txt", mse_log)
    write_log_to_file("ssim_log.txt", ssim_log)

    write_log_to_file("discriminator_loss_log.txt", discriminator_loss_log)
    write_log_to_file("generator_loss_log.txt", generator_loss_log)

    torch.save(generator.state_dict(), "generator.pt")
    torch.save(discriminator.state_dict(), "discriminator.pt")
    torch.save(optimizerG.state_dict(), "optimG.pt")
    torch.save(optimizerD.state_dict(), "optimD.pt")

    last_save_meta_data_file.write("Done => epoch: " + str(i) + "/" + str(epochs) + "; batch: " + str(j) + "/" + str(len(train_dataloader)) + "; D_loss=" + str(discriminator_loss.item()) + "; G_loss=" + str(generator_loss.item()))