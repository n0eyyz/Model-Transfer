import torch
from PIL import Image
import cv2
import numpy as np

# 1. 모델 로드 (첫 실행시만 모델 자동 다운로드)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 2. 이미지 로드 (경로 바꿔서 사용)
img_path = 'test.jpg'
img = Image.open(img_path)
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# 3. 탐지 실행
results = model(img)

# 4. 박스, 클래스, 확률로 결과 그리기
for *box, conf, cls in results.xyxy[0]:
    x1, y1, x2, y2 = map(int, box)
    label = f"{model.names[int(cls)]} {conf:.2f}"
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(img_cv, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)

# 5. 결과 저장
output_path = "detected_test.jpg"
cv2.imwrite(output_path, img_cv)
print(f"저장완료: {output_path}")
