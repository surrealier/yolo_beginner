import torch
import torch.nn as nn
import torch.nn.functional as F

def yolo_loss(outputs, labels, num_classes):
    # 예측된 바운딩 박스의 위치
    pred_bboxes = outputs[:, :4]
    pred_conf = outputs[:, 4]
    pred_class_probs = outputs[:, 5:5 + num_classes]  # 클래스 확률 예측

    # 실제 바운딩 박스와 객체 확률
    true_bboxes = labels[:, :, 1:5]
    true_conf = labels[:, :, 0]
    true_class_labels = labels[:, :, 5:]  # 실제 클래스 레이블

    # Localization loss (MSE)
    loc_loss = nn.MSELoss()(pred_bboxes, true_bboxes.mean(dim=1))
    
    # Confidence loss (MSE)
    conf_loss = nn.MSELoss()(pred_conf, true_conf.mean(dim=1))
    
    # Classification loss (CrossEntropy)
    class_loss = nn.CrossEntropyLoss()(pred_class_probs, true_class_labels.argmax(dim=2))
    
    # 최종 loss는 localization, confidence, classification loss의 합
    total_loss = loc_loss + conf_loss + class_loss
    
    return total_loss