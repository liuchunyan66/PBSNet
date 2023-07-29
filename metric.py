# from models.TestModel.MARSNet import BiSeNet

from torchstat import stat
import torchvision.models as models

model = models.resnet152()
stat(model, (3, 224, 224))

# model = BiSeNet(backbone='STDCNet1446', n_classes=19,
#                   use_boundary_2=False, use_boundary_4=False,
#                   use_boundary_8=False, use_boundary_16=False,
#                   use_conv_last=False)

