# pip install transformers pillow torch

from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch, os

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

# 이미지 폴더 내 임베딩
img_folder = "./"
image_files = [f for f in os.listdir(img_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
img_embeddings = []
for file in image_files:
    image = Image.open(os.path.join(img_folder, file)).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    img_emb = model.get_image_features(**inputs)
    img_embeddings.append((file, img_emb))

# 유저 설명
query = "누워 있는 고양이"
inputs = processor(text=[query], return_tensors="pt")
text_emb = model.get_text_features(**inputs)

# 유사도 계산
import torch.nn.functional as F
max_sim, best_img = -1, ""
for file, img_emb in img_embeddings:
    sim = F.cosine_similarity(img_emb, text_emb).item()
    if sim > max_sim:
        max_sim, best_img = sim, file

print(f"설명 '{query}'에 가장 어울리는 이미지는: {best_img} (유사도 {max_sim:.2f})")
