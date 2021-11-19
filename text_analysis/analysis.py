import jieba
import jieba.analyse
from jieba import analyse
textrank = analyse.textrank  #引入jieba中的TextRank
# 待分词的文本路径
sourceTxt = 'source.txt'
# 分好词后的文本路径
targetTxt = 'target.txt'

# 对文本进行操作
with open(sourceTxt, 'r', encoding = 'utf-8') as sourceFile, open(targetTxt, 'a+', encoding = 'utf-8') as targetFile:
    for line in sourceFile:
        seg = jieba.cut(line.strip(), cut_all = False)
        # 分好词之后之间用空格隔断
        output = ' '.join(seg)
        targetFile.write(output)
        targetFile.write('\n')
    print('写入成功！')

# 使用TF-IDF提取关键词
with open(targetTxt, 'r', encoding = 'utf-8') as file:
    text = file.readlines()
    """
    几个参数解释：
        * text : 待提取的字符串类型文本
        * topK : 返回TF-IDF权重最大的关键词的个数，默认为20个
        * withWeight : 是否返回关键词的权重值，默认为False
        * allowPOS : 包含指定词性的词，默认为空
    """
    keywords1 = jieba.analyse.extract_tags(str(text), topK = 10, withWeight=True, allowPOS=())
    print(keywords1)
    print('TF-IDF提取完毕！')


#使用Textrank进行关键词提取
    """
    参数解释：
        * text : 待提取的字符串类型文本
        * topK : 返回TextRank权重最大的关键词的个数，默认为20个
        * withWeight : 是否返回关键词的权重值，默认为False
        * allowPOS : 包含指定词性的词，默认为('ns', 'n', 'vn', 'v')
                     即仅提取地名、名词、动名词、动词
    """
    keywords2 = textrank(str(text),topK=5, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
    print(keywords2)
    print('TextRank提取完毕！')
	