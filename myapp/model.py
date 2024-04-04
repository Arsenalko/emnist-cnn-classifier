# your code here
import os
import torch
import torch.nn as nn
import numpy as np
import cv2

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.activation = nn.ReLU()
        self.conv_1 = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv_2 = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        y = self.activation(self.bn1(self.conv_1(x)))
        y = self.bn2(self.conv_2(y))
        return self.activation(x + y)

class EMNIST_CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv_1 = nn.Conv2d(1, 16, 3, 1)
        self.conv_2 = nn.Conv2d(16, 32, 3, 1)
        self.block_1 = ConvBlock(32, 32)
        self.pooling_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block_2 = ConvBlock(32, 32)
        self.pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.activation = nn.ReLU()
        self.flat = nn.Flatten()
        self.linear_1 = nn.Linear(1152, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.linear_2 = nn.Linear(512, 47)

    def forward(self, x):
        y = self.conv_1(x) 
        y = self.conv_2(y) 
        y = self.pooling_1(self.block_1(y)) 
        y = self.pooling_2(self.block_2(y)) 

        y = self.flat(y)
        y = self.activation(self.linear_1(y))
        y = self.bn1(y)
        y = self.activation(self.linear_2(y))
        return y

class Model:
    def __init__(self):
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        model_path = os.path.join('myapp', 'model.ckpt')
        self.model_ckpt = torch.load(model_path, map_location=torch.device(self.device))
        self.model = EMNIST_CNN()
        self.model.load_state_dict(self.model_ckpt['model_state_dict'])

    def predict(self, x):
        label_mapping = {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5',
            6: '6',
            7: '7',
            8: '8',
            9: '9',
            10: 'A',
            11: 'B',
            12: 'C',
            13: 'D',
            14: 'E',
            15: 'F',
            16: 'G',
            17: 'H',
            18: 'I',
            19: 'J',
            20: 'K',
            21: 'L',
            22: 'M',
            23: 'N',
            24: 'O',
            25: 'P',
            26: 'Q',
            27: 'R',
            28: 'S',
            29: 'T',
            30: 'U',
            31: 'V',
            32: 'W',
            33: 'X',
            34: 'Y',
            35: 'Z',
            36: 'a',
            37: 'b',
            38: 'd',
            39: 'e',
            40: 'f',
            41: 'g',
            42: 'h',
            43: 'n',
            44: 'q',
            45: 'r',
            46: 't'
        }

        self.model.eval()

        x = x.numpy()

        # Предобработка входного изображения (расширение, отражение по вертикали, поворот)
        kernel = np.ones((2,2), np.uint8)
        x = x.astype('uint8') 
        x = cv2.dilate(x, kernel, iterations=1)
        x = cv2.rotate(x[:,::-1], cv2.ROTATE_90_COUNTERCLOCKWISE)

        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32) / 255.0
        x = torch.from_numpy(x)
        x = x.to(self.device)

        self.model = self.model.to(self.device)
        pred = self.model(x)
        predicted_class = torch.nn.functional.softmax(pred.cpu(), dim=1).detach().numpy().argmax()
        return label_mapping[predicted_class]