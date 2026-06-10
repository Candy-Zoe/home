# Hugging Face Transformers文本摘要学习
# 主要内容：使用预训练模型进行文本摘要

from transformers import pipeline

print("=== 文本摘要 ===")
summarizer = pipeline('summarization')

text = """
Hugging Face is a company that develops tools for building applications using 
machine learning. They are best known for their Transformers library, which 
provides thousands of pre-trained models for natural language processing tasks. 
These models can be used for tasks such as text classification, sentiment analysis, 
text generation, question answering, and more. The library supports multiple 
frameworks including PyTorch, TensorFlow, and JAX.
"""

summary = summarizer(text, max_length=50, min_length=20)
print(f"原始文本:\n{text}")
print(f"\n摘要:\n{summary[0]['summary_text']}")

print("\n=== 中文文本摘要 ===")
chinese_summarizer = pipeline('summarization', model='uer/bart-large-chinese-cluecorpussmall')

chinese_text = """
人工智能是计算机科学的一个分支，它致力于研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。
深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的表示。深度学习在图像识别、语音识别、
自然语言处理等领域取得了巨大成功。PyTorch和TensorFlow是目前最流行的深度学习框架。
"""

summary = chinese_summarizer(chinese_text, max_length=50, min_length=20)
print(f"原始文本:\n{chinese_text}")
print(f"\n摘要:\n{summary[0]['summary_text']}")