# Transformers多模态学习
# 主要内容：多模态模型、图像字幕生成、视觉问答、文档理解

from transformers import pipeline, AutoProcessor, AutoModelForVision2Seq, AutoModel
import torch

print("=== 图像字幕生成 ===")
image_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

test_image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat.jpeg"
result = image_captioner(test_image_url)
print(f"图像描述: {result}")

print("\n=== 零样本图像分类 ===")
from transformers import pipeline

classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-large-patch14")

candidate_labels = ["cat", "dog", "car", "person"]
result = classifier(test_image_url, candidate_labels=candidate_labels)
print("零样本分类结果:")
for item in result:
    print(f"  {item['label']}: {item['score']:.4f}")

print("\n=== 文档问答 ===")
doc_qa = pipeline("document-question-answering", model="impira/layoutlm-document-qa")

print("\n=== 视觉问答 ===")
vqa_pipeline = pipeline("vqa-card")

print("\n=== 多模态嵌入 ===")
from transformers import AutoProcessor, AutoModel

model_name = "openai/clip-vit-base-patch32"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

print("多模态模型已加载")

print("\n=== 特征提取 ===")
from PIL import Image

image = Image.open("path_to_image.jpg").convert("RGB") if False else None
text = ["a cat", "a dog", "a car"]

print("图像和文本特征提取已准备")

print("\n=== 相似度计算 ===")
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    
    image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
    text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
    
    similarity = (image_embeds @ text_embeds.T).softmax(dim=-1)
    print(f"相似度分数: {similarity}")

print("\n=== 自定义多模态模型 ===")
from transformers import AutoModel

class MultimodalClassifier(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.vision_model = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.vision_model.config.hidden_size, 10)
    
    def forward(self, images):
        outputs = self.vision_model(**images)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

print("自定义多模态分类器已定义")