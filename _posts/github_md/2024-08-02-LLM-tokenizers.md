---
layout:     post
title:      "LLM的常见分词器-Tokenizer"
subtitle:   "常用Tokenizer"
date:       2024-08-02 18:36:00
author:     "CS"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Tokenizer
    - LLM
---

Tokenizer
--------------

可以先试玩 huggingface 上的 Tokenizer 演示 Demo : [Tokenizer playground](https://huggingface.co/spaces/Xenova/the-tokenizer-playground)

> 在nlp领域中tokenizer主要用于文本的预处理，能够将句子级文本转化为词级的文本，然后用于接下来的`词向量`转化，这个过程可以叫他token转化，或者直接叫tokenizer。
- 是LLM必不可少的一个部分，是语言模型最基础的组件。

1.  根据不同的切分粒度可以把tokenizer分为: **词级**（`Word-based`）,**字母级**（`Character-based`）和**子词级**（`Subword-based`），目前主要流行的方法是**子词级**转化。
2.  **subword**的切分包括: `BPE(/BBPE)`, `WordPiece` 和 `Unigram`三种分词模型。其中WordPiece可以认为是一种特殊的BPE。
3.  完整的分词流程包括：文本归一化，预切分，基于分词模型的切分，后处理。
4.  SentencePiece是一个分词工具，内置BEP等多种分词方法，基于Unicode编码并且将空格视为特殊的token。是当前大模型的主流分词方案。


基本的流程如图所示，包括归一化，预分词，基于分词模型的切分，后处理4个步骤。

![Tokenizer](http://i-blog.csdnimg.cn/blog_migrate/e4f1b15c311418f1f81532c10695b0f6.png)

#### **1. 归一化**

这是最基础的文本清洗，包括删除多余的换行和空格，转小写，移除音调等。例如：

```
input: Héllò hôw are ü?
normalization: hello how are u?
```

HuggingFace tokenizer的实现：[https://huggingface.co/docs/tokenizers/api/normalizers](https://huggingface.co/docs/tokenizers/api/normalizers)

#### **2. 预分词**

预分词阶段会把句子切分成更小的“词”单元。可以基于空格或者标点进行切分。 不同的tokenizer的实现细节是不一样的。例如:

```
input: Hello, how are you?

pre-tokenize:
[BERT]: [('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]

[GPT2]: [('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), ('Ġ', (14, 15)), ('Ġyou', (15, 19)), ('?', (19, 20))]

[t5]: [('▁Hello,', (0, 6)), ('▁how', (7, 10)), ('▁are', (11, 14)), ('▁you?', (16, 20))] 
```

可以看到BERT的tokenizer就是直接基于空格和标点进行切分。 
GPT2也是基于空格和标签，但是空格会保留成特殊字符“Ġ”。 
T5则只基于空格进行切分，标点不会切分。并且空格会保留成特殊字符"▁"，并且句子开头也会添加特殊字符"▁"。

HuggingFace tokenizer的实现： https://huggingface.co/docs/tokenizers/api/pre-tokenizers

#### **3. 基于分词模型的切分**

这里指的就是不同分词模型具体的切分方式。分词模型包括：BPE，WordPiece 和 Unigram 三种分词模型。

HuggingFace tokenizer的实现： https://huggingface.co/docs/tokenizers/api/models

#### **4. 后处理**

后处理阶段会包括一些特殊的分词逻辑，例如添加sepcial token：\[CLS\],\[SEP\]等。 HuggingFace tokenizer的实现： https://huggingface.co/docs/tokenizers/api/post-processors


### 三种不同分词粒度的Tokenizer

#### 1. Word-based

基于词的切分，会造成:
*   词表规模过大(英文单词过多)
*   会存在UNK(Unknown)，当限制词表的大小时，会存在很多不同的词都被置为UNK，导致信息丢失的问题。
 
```
input: Don't you?
Tokenize: ["Don", "'", "t", "you", "?"] ---期望--> ["Do", "n't", "you", "?"]
```  
不能学习到词缀之间的关系，例如：dog与dogs，happy与unhappy


#### 2. Character-based

基于字的切分，会造成:
*   每个token的信息含义、密度低
*   每个字拆分后，序列过长，输入Transformer的时候，可能会有长度限制，解码效率很低

```
input: Don't you?
Tokenize: ["D", "o", "n", "'", "t", "y","o", "u", "?"]  (英文只需用256个token词表，来表示所有字母、标点等字符)
```  

#### 3. Subword-based

所以基于词和基于字的切分方式是两个极端，其优缺点也是互补的。而折中的subword就是一种相对平衡的方案。

subword的基本切分原则是：
*   高频词依旧切分成完整的整词
*   低频词被切分成有意义的子词，例如 \[dogs => dog, s\]  \[tokenization => token, ization\]

基于subword的切分可以实现：
*   词表规模适中，解码效率较高
*   不存在UNK，信息不丢失
*   能学习到词缀之间的关系

### 三种常用的subword Tokenizer

|  |  |
| --- | --- |
| BPE(/BBPE) | GPT, GPT-2, GPT-J, GPT-Neo, RoBERTa, BART, LLaMA, ChatGLM-6B, Baichuan |
| WordPiece | BERT, DistilBERT，MobileBERT |
| Unigram | AlBERT, T5, mBART, XLNet |
| --- | --- |

Byte-Pair Encoding(BPE)是最广泛采用的subword分词器。
包含两个部分：**词频统计** 与 **词表合并**。

*   训练方法：从字符级的小词表出发，训练产生合并规则以及一个词表
*   编码方法：将文本切分成字符，再应用训练阶段获得的合并规则
*   经典模型：GPT, GPT-2, RoBERTa, BART, LLaMA, ChatGLM等


#### 1.BPE(/BBPE)

Byte-Pair Encoding(BPE)是最广泛采用的subword分词器。

*   词频统计：一般采用Word-based的Tokenizer进行统计。例如，\[("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs, 3")\]
*   文本切分成字符: 例如，\[("h", "u", "g", 10), ("p", "u", "g", 5), ("p", "u", "n", 12), ("b", "u", "n", 4), ("h", "u", "g", "s", 3)\]
*   相邻token合并: 例如，\[hu:15, ug:20, pu:17, un:16, bu:4, gs:5\]，可以指定合并次数
*   训练模型：GPT, GPT-2, RoBERTa, BART, LLaMA, ChatGLM等

**缺点**：
*   词表规模较大（中文等Unicode编码也被视为基本字符），解码效率较低

改进Byte-level BPE（BBPE）的方法是:
*   将字节视为基本Token
*   两个字节合并即为Unicode编码(中文等)

#### 2.WordPiece
WordPiece可以看作是BPE的变种。通过概率来保留词表中的词。
与BPE相比，WordPiece不会选择最频繁的符号对，而是会选择一种将训练数据添加到词汇表中的可能性最大化的符号对。参考[WordPiece](https://blog.csdn.net/weixin_42167712/article/details/110727139)

#### 3.Unigram

通过删除某一个对整体词表完整性影响小的组合，获取目标大小的词表。与BPE或WordPiece相比，Unigram将其基本词汇表初始化为大量符号，并逐步修整每个符号以获得较小的词汇表。 基本词汇可以是所有预先分词后的单词和最常见的子字符串。 Unigram不能直接用于Transformer中的任何模型，但可以与SentencePiece结合使用。
.


使用Tokenizer示例代码：
```python
# pip install transformers
from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
encoded_input = tokenizer(text)
print(encoded_input)
```

