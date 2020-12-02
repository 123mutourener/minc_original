import argparse
from torchvision import models


class PrintNetList(argparse.Action):
    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super(PrintNetList, self).__init__(option_strings, dest, nargs,
                                           **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print("Network implementations from the Torchvision module:")
        head = "COMMAND               NETWORK"
        print(head)
        net_col = len(head) - len("NETWORK")
        cmds = ["alexnet", "vgg11", "vgg11_bn", "vgg13",
                "vgg13_bn", "vgg16", "vgg16_bn", "vgg19",
                "vgg19_bn", "resnet18", "resnet34",
                "resnet50", "resnet101", "resnet152",
                "squeezenet1_0", "squeezenet1_1", "densenet121",
                "densenet161", "densenet169", "densenet201",
                "inception_v3", "googlenet", "shufflenet_v2_x1_0",
                "mobilenet_v2", "resnext50_32x4d", "resnext101_32x8d",
                "wide_resnet50_2", "wide_resnet101_2", "mnasnet1_0"]
        nets = ["AlexNet", "VGG 11-layers", "VGG 11-layers with batch norm.",
                "VGG 13-layers", "VGG 13-layers with batch norm.",
                "VGG 16-layers", "VGG 16 layers with batch norm.",
                "VGG 19-layers", "VGG 19-layers with batch norm.",
                "ResNet-18", "ResNet-34", "ResNet-50", "ResNet-101",
                "ResNet-152", "SqueezeNet 1.0", "SqueezeNet 1.1",
                "DenseNet-121 (BC)", "DenseNet-161 (BC)",
                "DenseNet-169 (BC)", "DenseNet-201 (BC)",
                "Inception v3", "GoogleNet", "ShuffleNet V2",
                "MobileNet V2", "ResNeXt-50-32x4d", "ResNeXt-101-32x8d",
                "Wide ResNet-50-2", "Wide ResNet-101-2", "MNASNet 1.0"]

        for cmd, net in zip(cmds, nets):
            print(cmd, end='')
            for char in range(net_col - len(cmd)):
                print(" ", end='')
            print(net)

        setattr(namespace, self.dest, True)


def torchvision_model(name, pretrained=True, num_classes=-1, **kwargs):
    net_builder = getattr(models, name)
    if pretrained:
        net = net_builder(pretrained=pretrained, **kwargs)
    elif num_classes > 0:
        net = net_builder(pretrained=pretrained, num_classes=num_classes, **kwargs)
    else:
        net = net_builder(pretrained=pretrained, **kwargs)

    return net


def get_model(model_name, pretrained=True, num_class=-1, **kwargs):
    net = torchvision_model(model_name, pretrained, num_class, **kwargs)

    return net


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model parsing module')

    parser.add_argument('--net-list', nargs=0, action=PrintNetList,
                        help='Print the list of the available network' +
                        'architectures')

    args = parser.parse_args()
