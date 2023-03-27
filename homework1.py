# -*- coding: utf-8 -*-

from collections import Counter
import os
import jieba
import math
basepath = 'C:\\Users\\HUAWEI\\Desktop\\深度学习与自然语言处理\\第一次作业'
with open(basepath+'\\jyxstxtqj_downcc.com\\inf.txt','r',encoding='ANSI') as f:
    booklist = f.read().split(',') #读目录 
with open(basepath+"\\cn_stopwords.txt",'r',encoding = 'utf-8-sig') as f:
    stopwords = f.read().split() #读停用词
#stopwords = set(stopwords)

stopwords2 = [' ','\n','本书来自www.cr173.com免费txt小说下载站','更多更新免费电子书请关注www.cr173.com','\u3000','目录']


def get_wf(words):
    twowords_fre = {}
    threewords_fre = {}
    for i in range(len(words)-1):
        twowords_fre[(words[i],words[i+1])] = twowords_fre.get((words[i],words[i+1]),0)+1
    for i in range(len(words)-2):
        threewords_fre[(words[i],words[i+1],words[i+2])] = threewords_fre.get((words[i],words[i+1],words[i+2]),0)+1
    return twowords_fre,threewords_fre
        
for i in booklist:
    #data_txt = read_data(basepath+'\\jyxstxtqj_downcc.com\\'+i+'.txt')
    path = basepath+'\\jyxstxtqj_downcc.com\\'+i+'.txt'
    with open(path, 'r', encoding='ANSI') as f:
        data_txt = f.read()
    bookname = i
    for i in stopwords2:
        data_txt = data_txt.replace(i,'')#去除停用词二
    lettercount = Counter() #统计词频
    letter_num = 0
    letters = []
    
    entropy_oneletter = 0  #每个字的一元模型信息熵
    entropy_twoletters = 0   #每个字的二元模型信息熵
    entropy_threeletters = 0   #每个字的三元模型信息熵

    entropy_oneword = 0 #每个词的一元模型信息熵
    entropy_twowords = 0  #每个词的二元模型信息熵
    entropy_threewords = 0  #每个词的三元模型信息熵
    
    for letter in data_txt:
        if letter not in stopwords:
            letters.append(letter)
            lettercount[letter] += 1
            letter_num += 1
    for i in lettercount.most_common():
        px = i[1]/letter_num
        entropy_oneletter +=  (- px*math.log(px,2))
    #print(bookname,"的字的一元模型信息熵为",entropy_oneletter)
    
    two_letters_fre,thr_letters_fre = get_wf(letters)
    bigram_letter_len = sum([dic[1] for dic in two_letters_fre.items()])
    for bi_word in two_letters_fre.items():
        jp_xy = bi_word[1] / bigram_letter_len  # 计算联合概率p(x,y)
        cp_xy = bi_word[1] / lettercount[bi_word[0][0]]  # 计算条件概率p(x|y)
        entropy_twoletters +=( -jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
   # print(bookname,"的字的二元模型信息熵为",entropy_twoletters)
    
    trigram_letter_len = sum([dic[1] for dic in thr_letters_fre.items()])
    for tri_word in thr_letters_fre.items():
        jp_xy = tri_word[1] / trigram_letter_len  # 计算联合概率p(x,y)
        cp_xy = tri_word[1] / two_letters_fre[tri_word[0][0:2]]  # 计算条件概率p(x|y)
        entropy_threeletters +=( -jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
    #print(bookname,"的字的三元模型信息熵为",entropy_threeletters)
    
    wordcount = Counter()
    word_num = 0
    words = []
    
    for word in jieba.cut(data_txt):
        
        if  word not in stopwords: #and len(word)>1 :
            words.append(word)
            wordcount[word] += 1
            word_num += 1
    for i in wordcount.most_common():
        px = i[1]/word_num
        entropy_oneword += (- px*math.log(px,2))
    #print(bookname,"的词的一元模型信息熵为",entropy_oneword)

    two_words_fre,thr_words_fre = get_wf(words)
    
    bigram_word_len = sum([dic[1] for dic in two_words_fre.items()])
    for bi_word in two_words_fre.items():
        jp_xy = bi_word[1] / bigram_word_len  # 计算联合概率p(x,y)
        cp_xy = bi_word[1] / wordcount[bi_word[0][0]]  # 计算条件概率p(x|y)
        entropy_twowords +=( -jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
    #print(bookname,"的词的二元模型信息熵为",entropy_twowords)
    
    trigram_word_len = sum([dic[1] for dic in thr_words_fre.items()])
    for tri_word in thr_words_fre.items():
        jp_xy = tri_word[1] / trigram_word_len  # 计算联合概率p(x,y)
        cp_xy = tri_word[1] / two_words_fre[tri_word[0][0:2]]  # 计算条件概率p(x|y)
        entropy_threewords +=( -jp_xy * math.log(cp_xy, 2))  # 计算三元模型的信息熵
    #print(bookname,"的词的三元模型信息熵为",entropy_threewords)
    
    print(bookname,entropy_oneletter,entropy_twoletters,entropy_threeletters)
    #print(bookname,entropy_oneword,entropy_twowords,entropy_threewords)
    
# 统计所有文章的整合
data_txt = ''
for i in booklist:
    #data_txt = read_data(basepath+'\\jyxstxtqj_downcc.com\\'+i+'.txt')
    path = basepath+'\\jyxstxtqj_downcc.com\\'+i+'.txt'
    with open(path, 'r', encoding='ANSI') as f:
        data_txt += f.read()
    
for i in stopwords2:
    data_txt = data_txt.replace(i,'')#去除停用词二
lettercount = Counter() #统计词频
letter_num = 0
letters = []

entropy_oneletter = 0  #每个字的一元模型信息熵
entropy_twoletters = 0   #每个字的二元模型信息熵
entropy_threeletters = 0   #每个字的三元模型信息熵

entropy_oneword = 0 #每个词的一元模型信息熵
entropy_twowords = 0  #每个词的二元模型信息熵
entropy_threewords = 0  #每个词的三元模型信息熵

for letter in data_txt:
    if letter not in stopwords:
        letters.append(letter)
        lettercount[letter] += 1
        letter_num += 1
for i in lettercount.most_common():
    px = i[1]/letter_num
    entropy_oneletter +=  (- px*math.log(px,2))
print("总的字的一元模型信息熵为",entropy_oneletter)

two_letters_fre,thr_letters_fre = get_wf(letters)

bigram_letter_len = sum([dic[1] for dic in two_letters_fre.items()])
for bi_word in two_letters_fre.items():
    jp_xy = bi_word[1] / bigram_letter_len  # 计算联合概率p(x,y)
    cp_xy = bi_word[1] / lettercount[bi_word[0][0]]  # 计算条件概率p(x|y)
    entropy_twoletters +=( -jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
print("总的字的二元模型信息熵为",entropy_twoletters)

trigram_letter_len = sum([dic[1] for dic in thr_letters_fre.items()])
for tri_word in thr_letters_fre.items():
    jp_xy = tri_word[1] / trigram_letter_len  # 计算联合概率p(x,y)
    cp_xy = tri_word[1] / two_letters_fre[tri_word[0][0:2]]  # 计算条件概率p(x|y)
    entropy_threeletters +=( -jp_xy * math.log(cp_xy, 2))  # 计算三元模型的信息熵
print("总的字的三元模型信息熵为",entropy_threeletters)

wordcount = Counter()
word_num = 0
words = []

for word in jieba.cut(data_txt):

    if  word not in stopwords: #and len(word)>1 :
        words.append(word)
        wordcount[word] += 1
        word_num += 1
for i in wordcount.most_common():
    px = i[1]/word_num
    entropy_oneword += (- px*math.log(px,2))
print("总的词的一元模型信息熵为",entropy_oneword)

two_words_fre,thr_words_fre = get_wf(words)

bigram_word_len = sum([dic[1] for dic in two_words_fre.items()])
for bi_word in two_words_fre.items():
    jp_xy = bi_word[1] / bigram_word_len  # 计算联合概率p(x,y)
    cp_xy = bi_word[1] / wordcount[bi_word[0][0]]  # 计算条件概率p(x|y)
    entropy_twowords +=( -jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
print("总的词的二元模型信息熵为",entropy_twowords)

trigram_word_len = sum([dic[1] for dic in thr_words_fre.items()])
for tri_word in thr_words_fre.items():
    jp_xy = tri_word[1] / trigram_word_len  # 计算联合概率p(x,y)
    cp_xy = tri_word[1] / two_words_fre[tri_word[0][0:2]]  # 计算条件概率p(x|y)
    entropy_threewords +=( -jp_xy * math.log(cp_xy, 2))  # 计算三元模型的信息熵
print("总的词的三元模型信息熵为",entropy_threewords)