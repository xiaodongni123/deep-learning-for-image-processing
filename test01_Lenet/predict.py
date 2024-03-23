# -*- coding:utf-8 -*-
import torch
import torchvision.transforms as transforms
from PIL import Image

from test01_Lenet.model import LeNet

def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth'))

    im = Image.open('1.png')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)    # [N, C, H, W]（因为一张照片的维度是没有batch的，所以要将batch加上）

    with torch.no_grad():
        outputs = net(im)
    #     predict = torch.max(outputs, dim=1)[1].numpy()  # 预测出最大概率的index，将其放入class类中，找到对应的类别
    # print(classes[int(predict)])

        predict = torch.softmax(outputs, dim=1)  # 预测的概率
    print(predict)



if __name__ == '__main__':
    main()
