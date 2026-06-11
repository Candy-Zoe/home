# NLTK文本处理基础学习
# 主要内容：文本分词、词性标注、停用词、词形归一化

# 导入NLTK库
import nltk

# 下载必要的语料库和模型
print("=== 下载NLTK资源 ===")
nltk.download('punkt', quiet=True)      # 句子分词器
nltk.download('averaged_perceptron_tagger', quiet=True)  # 词性标注器
nltk.download('stopwords', quiet=True)  # 停用词表
nltk.download('wordnet', quiet=True)    # 词形词典
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
print("资源下载完成")

# 文本分词
print("\n=== 文本分词 ===")

# 句子分词
text = "NLTK is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources."

# 使用sent_tokenize进行句子分词
sentences = nltk.sent_tokenize(text)
print("句子分词结果:")
for i, sent in enumerate(sentences, 1):
    print(f"  {i}. {sent}")

# 单词分词
words = nltk.word_tokenize(text)
print(f"\n单词分词结果: {words}")

# 词性标注
print("\n=== 词性标注 ===")

# 使用pos_tag进行词性标注
tagged_words = nltk.pos_tag(words)
print("词性标注结果:")
for word, tag in tagged_words[:10]:  # 只显示前10个
    print(f"  {word}: {tag}")

# 常用词性标签说明
print("\n常用词性标签:")
pos_tags = {
    'NN': '名词单数',
    'NNS': '名词复数',
    'VB': '动词原形',
    'VBD': '动词过去式',
    'VBG': '动词现在分词',
    'VBN': '动词过去分词',
    'JJ': '形容词',
    'JJR': '比较级形容词',
    'JJS': '最高级形容词',
    'RB': '副词',
    'IN': '介词',
    'DT': '限定词',
    'CC': '并列连词'
}
for tag, meaning in pos_tags.items():
    print(f"  {tag}: {meaning}")

# 停用词处理
print("\n=== 停用词处理 ===")

# 获取英文停用词列表
stop_words = set(nltk.corpus.stopwords.words('english'))
print(f"停用词表示例: {list(stop_words)[:10]}")

# 过滤停用词
text_words = nltk.word_tokenize("This is an example sentence showing how to remove stop words.")
filtered_words = [word for word in text_words if word.lower() not in stop_words]

print(f"\n原始文本: {text_words}")
print(f"过滤停用词后: {filtered_words}")

# 获取中文停用词
try:
    nltk.download('stopwords', quiet=True)
    chinese_stopwords = set(nltk.corpus.stopwords.words('chinese'))
    print(f"\n中文停用词表示例: {list(chinese_stopwords)[:5]}")
except:
    print("\n中文停用词未下载")

# 词形归一化
print("\n=== 词形归一化 ===")

# 词干提取（Stemming）- 使用Porter词干提取器
stemmer = nltk.PorterStemmer()
words_to_stem = ['running', 'runs', 'ran', 'runner', 'easily', 'studies', 'studying']

print("Porter词干提取器:")
for word in words_to_stem:
    print(f"  {word} -> {stemmer.stem(word)}")

# 使用Lancaster词干提取器
lancaster = nltk.LancasterStemmer()
print("\nLancaster词干提取器:")
for word in words_to_stem:
    print(f"  {word} -> {lancaster.stem(word)}")

# 词形还原（Lemmatization）- 使用WordNet词形还原器
lemmatizer = nltk.WordNetLemmatizer()
words_to_lemma = ['running', 'runs', 'ran', 'better', 'studies', 'geese']

print("\nWordNet词形还原器:")
for word in words_to_lemma:
    print(f"  {word} -> {lemmatizer.lemmatize(word)}")

# 指定词性进行词形还原
print("\n指定词性的词形还原:")
print(f"  'running' (默认): {lemmatizer.lemmatize('running')}")
print(f"  'running' (动词): {lemmatizer.lemmatize('running', pos='v')}")
print(f"  'better' (形容词): {lemmatizer.lemmatize('better', pos='a')}")
print(f"  'better' (名词): {lemmatizer.lemmatize('better', pos='n')}")

# n-gram模型
print("\n=== N-gram模型 ===")

# 创建bigram
from nltk import bigrams, trigrams

words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']

# 二元语法（bigram）
print("二元语法 (Bigrams):")
bg = list(bigrams(words))
print(f"  {bg}")

# 三元语法（trigram）
print("\n三元语法 (Trigrams):")
tg = list(trigrams(words))
print(f"  {tg}")

# 频率分布
print("\n=== 频率分布 ===")

# 统计词频
text = "The cat sat on the mat. The dog ran in the garden. The cat and dog played together."

words = nltk.word_tokenize(text.lower())
fdist = nltk.FreqDist(words)

print("词频统计:")
for word, count in fdist.most_common():
    print(f"  {word}: {count}")

# 查找最常见的词
print(f"\n最常见的3个词: {fdist.most_common(3)}")
print(f"词'the'出现次数: {fdist['the']}")

# 条件频率分布
print("\n=== 条件频率分布 ===")

# 准备条件文本数据
conditions = ['happy', 'sad', 'happy', 'sad', 'happy']
words_by_condition = [
    ['I', 'am', 'glad'],  # happy
    ['I', 'am', 'sad'],   # sad
    ['This', 'is', 'wonderful'],  # happy
    ['This', 'is', 'terrible'],  # sad
    ['Life', 'is', 'good']  # happy
]

cfd = nltk.ConditionalFreqDist(zip(conditions, words_by_condition))
print("条件频率分布:")
print(cfd.conditions())

# 特定条件的词汇
print(f"\nhappy条件下的词汇: {list(cfd['happy'])}")
print(f"sad条件下的词汇: {list(cfd['sad'])}")