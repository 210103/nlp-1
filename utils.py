from torch import random
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import json
from torchvision import transforms

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



class ImageCnn(nn.Module):
    def __init__(self, num_classes=100):
        super(ImageCnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class TextCnn(nn.Module):
    def __init__(self, num_classes=20):
        super(TextCnn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 1, stride=(1,2),kernel_size=(2,3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1,1,kernel_size=2,stride=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(8*8, num_classes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def read_img(path,question_nums=10000,img_h=224,img_w=224):
    image_list=torch.zeros(question_nums,3,img_h,img_w)
    transform=transforms.ToTensor()
    with open("./v2_OpenEnded_mscoco_train2014_questions.json", "r") as f:
        question_info = json.load(f)['questions'][0:question_nums]
        for i,j in enumerate(question_info):
            img_name=path+'/'+str(j['image_id'])+'.jpg'
            img=cv2.imread(img_name)
            img=cv2.resize(img,(img_h,img_w))
            image_list[i]=transform(img)
    return question_info,image_list

if __name__ == "__main__":
    model=TextCnn()
    print(model(torch.randn(2,1,10,20)).shape)