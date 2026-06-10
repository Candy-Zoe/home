# Hugging Face Transformers问答系统学习
# 主要内容：使用预训练模型进行问答

from transformers import pipeline

print("=== 问答系统 ===")
question_answerer = pipeline('question-answering')

context = """
Hugging Face is a company based in New York City that develops tools for 
building applications using machine learning. They are best known for their 
Transformers library, which provides thousands of pre-trained models for 
natural language processing tasks.
"""

questions = [
    "Where is Hugging Face based?",
    "What is Hugging Face known for?",
    "What does the Transformers library provide?"
]

for question in questions:
    result = question_answerer(question=question, context=context)
    print(f"问题: {question}")
    print(f"答案: {result['answer']}, 置信度: {result['score']:.4f}")
    print()

print("\n=== 中文问答 ===")
chinese_qa = pipeline('question-answering', model='uer/roberta-base-chinese-extractive-qa')

chinese_context = """
人工智能是计算机科学的一个分支，它致力于研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统。
深度学习是机器学习的一个子领域，它使用多层神经网络来学习数据的表示。
"""

chinese_questions = [
    "人工智能是什么？",
    "深度学习是什么？",
    "深度学习使用什么来学习？"
]

for question in chinese_questions:
    result = chinese_qa(question=question, context=chinese_context)
    print(f"问题: {question}")
    print(f"答案: {result['answer']}, 置信度: {result['score']:.4f}")
    print()