from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 1. 테스트할 이미지 경로 입력!
img_path = "fountain.jpeg"  # <-- 네 이미지 파일명으로 바꿔줘!

image = Image.open(img_path).convert('RGB')

# 2. BLIP 모델 불러오기 (첫 실행 땐 다운로드라 조금 느릴 수 있음)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 3. 캡션 생성
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
print("AI가 지은 제목:", caption)
