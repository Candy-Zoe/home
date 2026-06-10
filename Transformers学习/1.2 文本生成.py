# Hugging Face Transformers文本生成学习
# 主要内容：使用GPT模型进行文本生成

from transformers import pipeline

print("=== 文本生成 ===")
generator = pipeline('text-generation', model='gpt2')

prompt = "In the future, artificial intelligence will"
result = generator(prompt, max_length=50, num_return_sequences=2)

print(f"提示词: {prompt}")
print("\n生成结果:")
for i, output in enumerate(result):
    print(f"{i+1}. {output['generated_text']}")

print("\n=== 中文文本生成 ===")
chinese_generator = pipeline('text-generation', model='uer/gpt2-chinese-cluecorpussmall')

chinese_prompt = "人工智能正在改变世界，"
result = chinese_generator(chinese_prompt, max_length=50, num_return_sequences=2)

print(f"提示词: {chinese_prompt}")
print("\n生成结果:")
for i, output in enumerate(result):
    print(f"{i+1}. {output['generated_text']}")