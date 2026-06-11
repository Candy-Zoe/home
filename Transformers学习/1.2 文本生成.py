# Hugging Face Transformers文本生成学习
# 主要内容：GPT模型文本生成、文本补全、对话生成

# 导入必要的库
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# 使用pipeline进行文本生成
print("=== 使用pipeline进行文本生成 ===")

# 创建文本生成pipeline
generator = pipeline('text-generation', model='gpt2')

# 测试文本生成
prompt = "In the future, artificial intelligence will"

# 生成文本
results = generator(
    prompt,
    max_length=50,
    num_return_sequences=3,
    temperature=0.7,
    top_k=50,
    repetition_penalty=1.2
)

print(f"提示词: {prompt}")
print("\n生成结果:")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['generated_text']}")

# 使用中文模型
print("\n=== 使用中文GPT模型 ===")

# 创建中文文本生成pipeline
try:
    chinese_generator = pipeline('text-generation', model='uer/gpt2-chinese-cluecorpussmall')
    
    # 测试中文文本生成
    chinese_prompt = "人工智能正在改变世界，未来"
    
    chinese_results = chinese_generator(
        chinese_prompt,
        max_length=50,
        num_return_sequences=2,
        temperature=0.8,
        repetition_penalty=1.1
    )
    
    print(f"提示词: {chinese_prompt}")
    print("\n中文生成结果:")
    for i, result in enumerate(chinese_results, 1):
        print(f"{i}. {result['generated_text']}")
except Exception as e:
    print("中文模型加载失败，可能需要下载模型")
    print(f"错误信息: {e}")

# 使用自定义模型和tokenizer
print("\n=== 使用自定义模型 ===")

# 选择预训练模型
model_name = "gpt2"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer加载成功: {tokenizer.__class__.__name__}")

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"模型加载成功: {model.__class__.__name__}")

# 准备输入文本
text = "Machine learning is a subset of artificial intelligence"

# 对文本进行编码
inputs = tokenizer(text, return_tensors="pt")
print(f"\n输入编码:")
print(f"  input_ids形状: {inputs['input_ids'].shape}")
print(f"  token数量: {inputs['input_ids'].shape[1]}")

# 使用模型进行生成
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        repetition_penalty=1.2,
        do_sample=True
    )

# 解码输出
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n生成文本: {generated_text}")

# 控制生成参数
print("\n=== 控制生成参数 ===")

# 不同参数的效果对比
parameters = [
    {"temperature": 0.2, "top_k": 50, "description": "保守生成"},
    {"temperature": 0.7, "top_k": 50, "description": "平衡生成"},
    {"temperature": 1.5, "top_k": 50, "description": "创造性生成"},
]

print("不同参数生成效果对比:")
for params in parameters:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=70,
            num_return_sequences=1,
            temperature=params["temperature"],
            top_k=params["top_k"],
            repetition_penalty=1.2,
            do_sample=True
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n{params['description']} (temperature={params['temperature']}):")
    print(f"  {text}")

# top_p采样（核采样）
print("\n=== top_p采样 ===")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=70,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,  # 只考虑累积概率达到90%的token
        repetition_penalty=1.2,
        do_sample=True
    )
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"top_p=0.9生成结果:")
print(f"  {text}")

# 文本补全
print("\n=== 文本补全 ===")

# 创建文本补全pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')

# 测试文本补全
result = unmasker("Machine learning is a [MASK] of artificial intelligence.")
print("文本补全结果:")
for r in result:
    print(f"  {r['sequence']} (得分: {r['score']:.4f})")

# 对话生成
print("\n=== 对话生成 ===")

# 创建对话生成pipeline
chatbot = pipeline('conversational', model='microsoft/DialoGPT-small')

# 创建对话历史
from transformers import Conversation

# 示例对话
conversation = Conversation("Hello! How are you?")
print(f"用户: {conversation.past_user_inputs[-1]}")

# 生成回应
conversation = chatbot(conversation)
print(f"机器人: {conversation.generated_responses[-1]}")

# 继续对话
conversation.add_user_input("What is machine learning?")
print(f"\n用户: {conversation.past_user_inputs[-1]}")

conversation = chatbot(conversation)
print(f"机器人: {conversation.generated_responses[-1]}")

# 批处理文本生成
print("\n=== 批处理文本生成 ===")

# 多个提示词
prompts = [
    "The future of AI is",
    "Machine learning algorithms can",
    "Natural language processing allows"
]

# 批量编码
inputs = tokenizer(prompts, padding=True, return_tensors="pt")
print(f"批量输入形状: {inputs['input_ids'].shape}")

# 批量生成
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        repetition_penalty=1.2,
        do_sample=True
    )

# 解码多个输出
print("\n批量生成结果:")
for i, output in enumerate(outputs):
    text = tokenizer.decode(output, skip_special_tokens=True)
    print(f"{i+1}. {text}")

# 限制生成内容
print("\n=== 限制生成内容 ===")

# 设置特殊token来控制生成
# 示例：只允许生成特定词汇
allowed_tokens = ["machine", "learning", "AI", "intelligence", "algorithm", "data"]

# 获取token ID
allowed_ids = [tokenizer.encode(token)[0] for token in allowed_tokens]
print(f"允许的token: {allowed_tokens}")
print(f"对应的ID: {allowed_ids}")

# 使用force_words_ids强制生成特定词汇
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        repetition_penalty=1.2,
        do_sample=True,
        force_words_ids=[[tokenizer.encode(t)[0]] for t in allowed_tokens]
    )

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\n强制生成结果: {text}")

# 使用stop_token结束生成
print("\n=== 使用stop_token ===")

# 设置停止token
stop_token = tokenizer.encode(".")[0]

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        repetition_penalty=1.2,
        do_sample=True,
        eos_token_id=stop_token  # 使用句号作为结束符
    )

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"使用句号结束生成: {text}")

print("\n文本生成学习完成！")