import os # os 모듈을 임포트합니다. 이 모듈은 운영체제와 상호작용하기 위한 여러 기능을 제공합니다.
import torch # PyTorch를 임포트합니다. PyTorch는 파이썬 기반의 딥러닝 프레임워크로, GPU를 사용한 연산을 지원하며, 자동 미분(autograd)을 지원합니다.
from torch.utils.data import Dataset, DataLoader # torch.utils.data 모듈에서 Dataset과 DataLoader를 임포트합니다. Dataset은 데이터셋을 나타내는 추상 클래스이며, DataLoader는 데이터셋을 미니배치로 나누어 읽어오는 클래스입니다.
import cv2 # OpenCV를 임포트합니다. OpenCV는 영상 처리 라이브러리로, 이미지나 비디오 파일을 읽고 쓰는 기능을 제공합니다.
import numpy as np # NumPy를 임포트합니다. NumPy는 파이썬의 수치 계산 라이브러리로, 다차원 배열을 효과적으로 다룰 수 있습니다.

class SimpleDataset(Dataset): # SimpleDataset 클래스를 정의합니다. 이 클래스는 torch.utils.data.Dataset을 상속받아야 합니다. Dataset은 데이터셋을 나타내는 추상 클래스입니다.
    def __init__(self, data_dir, transform=None): # 생성자를 정의합니다. 생성자는 클래스의 인스턴스를 초기화하는 역할을 합니다. 인자로 data_dir와 transform을 받습니다. data_dir는 데이터셋이 저장된 디렉토리의 경로를 의미하며, transform은 데이터셋에 적용할 전처리 함수를 의미합니다.
        self.data_dir = data_dir # data_dir을 인스턴스 변수로 저장합니다.
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')] # data_dir 디렉토리에 있는 .jpg 파일들의 리스트를 생성합니다.
        self.transform = transform # transform을 인스턴스 변수로 저장합니다.

    def __len__(self): # __len__ 함수를 정의합니다. 이 함수는 데이터셋의 크기를 반환합니다. 즉, 데이터셋에 있는 샘플의 개수를 반환합니다.
        return len(self.image_files) # 데이터셋에 있는 샘플의 개수를 반환합니다.
    
    def __getitem__(self, idx): # __getitem__ 함수를 정의합니다. 이 함수는 주어진 인덱스에 해당하는 샘플을 반환합니다. 즉, 데이터셋에서 특정 인덱스에 해당하는 샘플을 가져옵니다.
        img_path = os.path.join(self.data_dir, self.image_files[idx]) # 데이터셋 디렉토리와 이미지 파일 이름을 합쳐 이미지 파일의 경로를 생성합니다.
        image = cv2.imread(img_path) # 이미지 파일을 읽어옵니다.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 이미지의 색상 채널을 RGB로 변경합니다.
        image = cv2.resize(image, (224, 224)) # 이미지의 크기를 224x224로 변경합니다.

        label = os.path.join(self.data_dir, self.image_files[idx].replace('.jpg', '.txt')) # 레이블 파일의 경로를 생성합니다. 이미지 파일의 확장자를 .txt로 변경합니다.
        label = np.loadtxt(label) # 레이블 파일을 읽어옵니다.

        if self.transform: # transform이 주어졌다면 
            image = self.transform(image) # 이미지에 transform을 적용합니다.
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0 # 이미지를 텐서로 변환합니다. 이미지의 shape은 (H, W, C)에서 (C, H, W)로 변경하고, 255로 나누어 정규화합니다. 왜냐하면 픽셀 값은 0부터 255 사이의 값을 가지기 때문입니다. dtype은 torch.float32로 설정합니다. float16은 너무 작은 값을 표현할 수 없기 때문에 사용하지 않습니다.
        label = torch.tensor(label, dtype=torch.float32) # 레이블을 텐서로 변환합니다. dtype은 torch.long으로 설정합니다. long은 정수형 데이터 타입 중 하나입니다. 

        return image, label # 이미지와 레이블을 반환합니다.

def collate_fn(batch):
    images, labels = list(zip(*batch))
    images = torch.stack(images)

    # 라벨을 패딩하여 텐서로 변환
    max_size = max(label.shape[0] for label in labels)

    padded_labels = []
    for label in labels:
        pad = torch.zeros(max_size - label.shape[0], label.shape[1])
        padded_labels.append(torch.cat([label, pad], dim=0))

    labels = torch.stack(padded_labels)

    return images, labels
    
def get_data_loader(data_dir, batch_size=16, shuffle=True): # get_data_loader 함수를 정의합니다. 이 함수는 DataLoader를 생성하여 반환합니다. DataLoader는 데이터셋을 미니배치로 나누어 읽어오는 클래스입니다. 이 함수가 호출되는 주기는 학습 루프에서 데이터를 읽어오는 주기와 일치합니다. 예를 들어, 에폭마다 호출됩니다.
    dataset = SimpleDataset(data_dir) # SimpleDataset을 생성합니다.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn) # DataLoader를 생성합니다. DataLoader는 데이터셋을 미니배치로 나누어 읽어오는 클래스입니다. 인자로는 데이터셋과 미니배치 크기, 셔플 여부를 받습니다.
    return loader # DataLoader를 반환합니다.