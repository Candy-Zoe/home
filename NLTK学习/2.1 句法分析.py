# NLTK句法分析学习
# 主要内容：句法分析、命名实体识别、语义分析、文本生成

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print("=== 下载必要资源 ===")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')

print("\n=== 分词和词性标注 ===")
text = "Natural language processing is a subfield of artificial intelligence."
tokens = word_tokenize(text)
tagged = pos_tag(tokens)
print(f"分词结果: {tokens}")
print(f"词性标注: {tagged}")

print("\n=== 命名实体识别 ===")
text = "Barack Obama was born in Hawaii. He was the 44th President of the United States."
tokens = word_tokenize(text)
tagged = pos_tag(tokens)
entities = ne_chunk(tagged)
print(f"命名实体识别结果:")
for subtree in entities.subtrees():
    if subtree.label() == 'PERSON':
        person = ' '.join(word for word, tag in subtree.leaves())
        print(f"  人名: {person}")
    elif subtree.label() == 'GPE':
        gpe = ' '.join(word for word, tag in subtree.leaves())
        print(f"  地点: {gpe}")

print("\n=== 停用词处理 ===")
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
print(f"原始词数: {len(tokens)}")
print(f"过滤后词数: {len(filtered_tokens)}")
print(f"过滤后词汇: {filtered_tokens}")

print("\n=== 词形还原 ===")
lemmatizer = WordNetLemmatizer()
words = ["running", "ran", "runs", "better", "best", "dogs", "dog"]
lemmatized = [lemmatizer.lemmatize(word) for word in words]
print(f"原始词汇: {words}")
print(f"词形还原: {lemmatized}")

print("\n=== 词干提取 ===")
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in words]
print(f"词干提取: {stemmed}")

print("\n=== 语法分析 ===")
grammar = """
NP: {<DT>?<JJ>*<NN>}
VP: {<VB.*><NP|PP>*}
"""
cp = nltk.RegexpParser(grammar)
result = cp.parse(tagged)
print(f"语法分析树:")
result.pretty_print()

print("\n=== 文本相似度 ===")
from nltk.corpus import wordnet

syn1 = wordnet.synsets('car')[0]
syn2 = wordnet.synsets('automobile')[0]
similarity = syn1.wup_similarity(syn2)
print(f"car 和 automobile 的相似度: {similarity:.4f}")