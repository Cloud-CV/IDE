import torchvision.models as models
import torch
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


resnet101 = models.resnet101()
squeezenet1_0 = models.squeezenet1_0()
resnet152 = models.resnet152()
torch.save(resnet101, dir_path + '/resnet101.pt')
torch.save(resnet152, dir_path + '/resnet152.pt')
torch.save(squeezenet1_0, dir_path + '/squeezenet1.pt')
