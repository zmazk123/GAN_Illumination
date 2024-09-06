import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.utils import save_image

from dataset import CustomImageDataset
from model import Generator


def denormalize(image):
    # denormalize [-1,1] -> [0,255]
    image = image.add_(1).div_(2).mul_(255)
    image = torch.clamp(image, 0, 255)
    image = torch.IntTensor(image.size()).copy_(image)

    return image

def show_image(image_tensor):
    y = denormalize(image_tensor)
    y = y.permute(1, 2, 0)
    plt.imshow(y)
    plt.show()

def save_image_my(image_tensor, image_name):
    y = image_tensor.add_(1).div_(2)
    save_image(y, image_name)


batch_size = 1
generator = Generator(12, 3, 128)
generator.load_state_dict(torch.load("generator.pt", map_location=torch.device('cpu')))
generator.eval()

test_data = CustomImageDataset("sample_dataset/test/")
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

mse = MeanSquaredError()
ssim = StructuralSimilarityIndexMeasure(data_range=(-1,1))

timearr = []
with torch.no_grad():
    for i, (image_depth, image_normal, image_diffuse, image_direct, image_global) in enumerate(test_dataloader):
        print(i)
        input = torch.cat((image_diffuse, image_direct, image_normal, image_depth), 1)
        fake_image = generator(input)

        mse.update(fake_image, image_global)
        ssim.update(fake_image, image_global)

        #show_image(image_global[0].clone().detach())
        #show_image(fake_image[0].clone().detach())
        #save_image_my(fake_image[0].clone().detach(), "test_output/"+str(i)+".png")

    print(mse.compute())
    mse.reset()
    print(ssim.compute())
    ssim.reset()