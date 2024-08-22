import torch # PyTorch를 임포트합니다. PyTorch는 파이썬 기반의 딥러닝 프레임워크로, GPU를 사용한 연산을 지원하며, 자동 미분(autograd)을 지원합니다.
import torch.nn as nn # pytorch의 neural network 모듈을 임포트합니다. 이는 파이토치의 모든 모듈들이 nn.Module을 상속받아 구현되어야 하기 때문입니다.
import torch.nn.functional as F # pytorch의 functional 모듈을 임포트합니다. 이 모듈은 nn.Module을 상속받지 않는 함수들을 포함하고 있습니다. 예를 들어 ReLU나 Sigmoid 함수 등이 있습니다.

class ConvBlock(nn.Module): # ConvBlock 클래스를 정의합니다. 이 클래스는 nn.Module을 상속받아야 합니다.
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding): # 생성자를 정의합니다. 생성자는 클래스의 인스턴스를 초기화하는 역할을 합니다.
        super(ConvBlock, self).__init__() # nn.Module의 생성자를 호출합니다. super() 함수를 사용하여 부모 클래스의 생성자를 호출할 수 있습니다. 인자로는 자식 클래스의 이름과 self를 전달합니다. 왜냐하면 파이썬에서는 부모 클래스의 생성자를 자식 클래스에서 직접 호출하지 않으면 부모 클래스의 생성자가 호출되지 않기 때문입니다.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) # nn.Conv2d를 사용하여 Convolutional 레이어를 생성합니다. 이 레이어는 in_channels개의 입력 채널을 받아 out_channels개의 출력 채널을 생성합니다.
        self.bn = nn.BatchNorm2d(out_channels) # Batch Normalization 레이어를 생성합니다. 이 레이어는 Convolutional 레이어의 출력을 정규화합니다. 예를 들어, 평균을 0, 분산을 1로 만듭니다.
        self.leaky_relu = nn.LeakyReLU(0.1) # LeakyReLU 함수를 생성합니다. LeakyReLU는 ReLU 함수의 변종으로, 입력이 음수일 때 작은 기울기를 갖습니다. 이는 ReLU 함수의 출력이 0이 되는 문제를 해결하기 위해 사용됩니다.

    def forward(self, x): # forward 함수를 정의합니다. 이 함수는 모델이 입력 데이터를 받아 순전파 연산을 수행하는 함수입니다.
        x = self.conv(x) # 입력 데이터 x를 Convolutional 레이어에 통과시킵니다.
        x = self.bn(x)  # Convolutional 레이어의 출력을 Batch Normalization 레이어에 통과시킵니다.
        x = self.leaky_relu(x) # Convolutional 레이어의 출력을 LeakyReLU 함수에 통과시킵니다.
        return x # 변환된 x를 반환합니다.
    
class SimpleYOLOv8(nn.Module): # SimpleYOLOv8 클래스를 정의합니다. 이 클래스는 nn.Module을 상속받아야 합니다.
    def __init__(self, num_classes=20): # 생성자를 정의합니다. 생성자는 클래스의 인스턴스를 초기화하는 역할을 합니다. 인자로 num_classes를 받습니다. num_classes는 클래스의 개수를 의미합니다.
        super(SimpleYOLOv8, self).__init__() # nn.Module의 생성자를 호출합니다. super() 함수를 사용하여 부모 클래스의 생성자를 호출할 수 있습니다. 인자로는 자식 클래스의 이름과 self를 전달합니다. 왜냐하면 파이썬에서는 부모 클래스의 생성자를 자식 클래스에서 직접 호출하지 않으면 부모 클래스의 생성자가 호출되지 않기 때문입니다.
        self.conv1 = ConvBlock(3, 16, 3, 1, 1) # ConvBlock을 사용하여 첫 번째 Convolutional 레이어를 생성합니다. 이 레이어는 3개의 입력 채널을 받아 16개의 출력 채널을 생성합니다. 3x3 커널을 사용하며, stride는 1, padding은 1입니다. 3개의 채널은 RGB 이미지의 채널 수입니다.
        self.conv2 = ConvBlock(16, 32, 3, 2, 1) # ConvBlock을 사용하여 두 번째 Convolutional 레이어를 생성합니다. 이 레이어는 16개의 입력 채널을 받아 32개의 출력 채널을 생성합니다. 3x3 커널을 사용하며, stride는 2, padding은 1입니다. 
        self.conv3 = ConvBlock(32, 64, 3, 2, 1) # ConvBlock을 사용하여 세 번째 Convolutional 레이어를 생성합니다. 이 레이어는 32개의 입력 채널을 받아 64개의 출력 채널을 생성합니다. 3x3 커널을 사용하며, stride는 2, padding은 1입니다.
        self.conv4 = ConvBlock(64, 128, 3, 2, 1) # ConvBlock을 사용하여 네 번째 Convolutional 레이어를 생성합니다. 이 레이어는 64개의 입력 채널을 받아 128개의 출력 채널을 생성합니다. 3x3 커널을 사용하며, stride는 2, padding은 1입니다.
        self.conv5 = ConvBlock(128, 256, 3, 2, 1) # ConvBlock을 사용하여 다섯 번째 Convolutional 레이어를 생성합니다. 이 레이어는 128개의 입력 채널을 받아 256개의 출력 채널을 생성합니다. 3x3 커널을 사용하며, stride는 2, padding은 1입니다.
        self.conv6 = ConvBlock(256, 512, 3, 2, 1) # ConvBlock을 사용하여 여섯 번째 Convolutional 레이어를 생성합니다. 이 레이어는 256개의 입력 채널을 받아 512개의 출력 채널을 생성합니다. 3x3 커널을 사용하며, stride는 2, padding은 1입니다.

        self.fc = nn.Linear(512 * 7 * 7, num_classes)  # Fully Connected 레이어를 생성합니다. 이 레이어는 512x7x7개의 입력을 받아 num_classes개의 출력을 생성합니다. 7x7은 입력 이미지의 크기가 224x224로 줄어든 것을 의미합니다. 만약 입력 이미지의 크기가 448x448이라면 14x14이 될 것입니다.

    def forward(self, x): # forward 함수를 정의합니다. 이 함수는 모델이 입력 데이터를 받아 순전파 연산을 수행하는 함수입니다.
        x = self.conv1(x) # 입력 데이터 x를 첫 번째 Convolutional 레이어에 통과시킵니다.
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(x.size(0), -1) # Convolutional 레이어의 출력을 1차원 벡터로 변환합니다. 이는 Fully Connected 레이어에 입력하기 위함입니다. 
        x = self.fc(x) # 1차원 벡터를 Fully Connected 레이어에 통과시킵니다. 이 레이어는 num_classes개의 출력을 생성합니다.

        return x # 변환된 x를 반환합니다. 이는 모델의 출력입니다. shape은 (batch_size, num_classes)가 될 것입니다.