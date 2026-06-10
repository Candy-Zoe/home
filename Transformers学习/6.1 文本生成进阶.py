# Transformers文本生成进阶学习
# 主要内容：GPT模型、文本补全、对话生成、控制生成

from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer, set_seed
import torch

print("=== 加载GPT-2模型 ===")
generator = pipeline('text-generation', model='gpt2')
set_seed(42)

print("\n=== 文本生成 ===")
prompt = "Once upon a time"
result = generator(prompt, max_length=50, num_return_sequences=1)
print(f"生成文本:\n{result[0]['generated_text']}")

print("\n=== 多个生成结果 ===")
results = generator("The future of AI is", max_length=30, num_return_sequences=3)
print("多个生成结果:")
for i, r in enumerate(results):
    print(f"  {i+1}. {r['generated_text']}")

print("\n=== 控制生成参数 ===")
result = generator(
    "In a world where",
    max_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True
)
print(f"控制生成结果:\n{result[0]['generated_text']}")

print("\n=== 不同温度对比 ===")
temperatures = [0.5, 1.0, 1.5]
prompt = "The meaning of life is"

for temp in temperatures:
    result = generator(prompt, max_length=30, temperature=temp, do_sample=True)
    print(f"温度={temp}: {result[0]['generated_text']}")

print("\n=== 文本补全 ===")
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

input_text = "The quick brown fox"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

output = model.generate(
    input_ids,
    max_length=50,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"束搜索生成:\n{generated_text}")

print("\n=== 对话生成 ===")
dialogue_prompt = "User: Hello, how are you?\nAssistant:"
result = generator(dialogue_prompt, max_length=50)
print(f"对话生成:\n{result[0]['generated_text']}")

print("\n=== 故事续写 ===")
story_prompt = "It was a dark and stormy night. The old house at the end of the street had been empty for years."
result = generator(story_prompt, max_length=100, temperature=0.8)
print(f"故事续写:\n{result[0]['generated_text']}")

print("\n=== 代码生成 ===")
code_prompt = "def fibonacci(n):"
result = generator(code_prompt, max_length=60, temperature=0.2)
print(f"代码生成:\n{result[0]['generated_text']}")

print("\n=== 问答生成 ===")
qa_prompt = "Question: What is machine learning?\nAnswer:"
result = generator(qa_prompt, max_length=50)
print(f"问答生成:\n{result[0]['generated_text']}")

print("\n=== 使用LogitsProcessor控制生成 ===")
from transformers import LogitsProcessor, LogitsProcessorList

class CustomLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        scores[:, tokenizer.encode('bad', add_special_tokens=False)[0]] = -float('inf')
        return scores

custom_processor = LogitsProcessorList([CustomLogitsProcessor()])
print("自定义Logits处理器已创建")

print("\n=== 流式生成 ===")
from transformers import TextIteratorStreamer
import threading

streamer = TextIteratorStreamer(tokenizer)
config = {'max_length': 50}

def generate():
    model.generate(input_ids, **config, streamer=streamer)

thread = threading.Thread(target=generate)
thread.start()

print("流式生成:")
for text in streamer:
    print(text, end='', flush=True)
print()

thread.join()