from transformers import CLIPModel
import torch

# 1. 네트워크 구조(모델 클래스) 정의 & 가중치 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
model.eval()

# 2. 더미 입력 (입력 shape 맞게)
dummy_input = torch.randn(1, 3, 224, 224)  # 예: 3채널 224x224 이미지

# 3. ONNX로 export
torch.onnx.export(model, dummy_input, "model.onnx", input_names=['input'], output_names=['output'], opset_version=13)