import torch
from models.simple_yolo import SimpleYOLOv8
from utils.dataset import get_data_loader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/val/', help='data directory')
parser.add_argument('--batch', type=int, default=16, help='batch size')
parser.add_argument('--model', type=str, default='yolov8_simple.pth', help='model path')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleYOLOv8(num_classes=20).to(device)
model.load_state_dict(torch.load(args.model))
model.eval()

val_loader = get_data_loader(args.data, batch_size=args.batch)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        confidence, predicted = torch.max(outputs.data, 1) # 가장 큰 값의 인덱스를 가져옵니다. _는 가장 큰 값이고, predicted는 가장 큰 값의 인덱스입니다. 따라서 _는 confidence score이고, predicted는 예측된 클래스입니다.
        total += labels.size(0) # 레이블의 총 개수를 더합니다. labels.size(0)은 레이블의 개수입니다. 
        correct += (predicted == labels).sum().item() # 정답의 개수를 더합니다. predicted == labels는 예측된 클래스와 실제 클래스가 같은지를 나타내는 Bool 텐서입니다. .sum()은 True의 개수를 세는 함수이고, .item()은 정수형 값을 가져오는 함수입니다. 따라서, 정답이면 1, 오답이면 0이 더해집니다.

print(f"Accuracy: {100 * correct / total:.2f}%")