import PyPDF2
import re
import csv
import nltk
from nltk.tag import CRFTagger
import pycrfsuite
from trainer import sent2features
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from difflib import SequenceMatcher

# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

ct = CRFTagger()
ct.set_model_file('all_indo_man_tag_corpus_model.crf.tagger')

def getOrg(raw_sent):
    ct = CRFTagger()
    ct.set_model_file('all_indo_man_tag_corpus_model.crf.tagger')
    
    tokens = nltk.tokenize.word_tokenize(raw_sent)
    postagged = ct.tag_sents([tokens])
    
    data = []
    for token in postagged[0]:
        data.append(token+('O',))
    
    tagger_ner = pycrfsuite.Tagger()
    tagger_ner.open('model_ner.crfsuite')
    ner = tagger_ner.tag(sent2features(data,False))
    
    for i in range(len(ner)):
        data[i] = data[i][0:2]+(ner[i],)
       
    
    org = []
    
    for token in data:
        label = token[2][-3:]
        if label == 'ORG':
            org.append(token[0])

    return (' ').join(org)

def clean_text(text):
    first = re.sub(r'[.,:]$', '', text.replace(' pada ',''))
    return first.split((re.findall(r' [a-z]', first) + ['      '])[0])[0]

def stem_it(sentence):
    res = []
    tokens = nltk.tokenize.word_tokenize(sentence)
    for token in tokens:
        output = stemmer.stem(token)
        res.append((token,output))
    return res

def postag_it(sentence, tokenized=False):
    tokens = []
    if tokenized:
        tokens = sentence
    else:
        tokens = nltk.tokenize.word_tokenize(sentence)
    postagged = ct.tag_sents([tokens])
    return postagged[0]

def concat_tokens(stem, pt):
    i = 0
    res = []
    for i in range(len(stem)):
        res.append([stem[i][0],stem[i][1],pt[i][1]])
    return res

def clean_pt(pt):
    res = []
    for pos in pt:
        if pos[2] in ['NN','NNP','VB', 'PRP']:
            res.append(pos)
    return res

def get_total_dist(source, keyword):
    maxx = 0
    for token in nltk.tokenize.word_tokenize(source):
        dist = SequenceMatcher(None, token, keyword).ratio()
        if dist > maxx:
            maxx = dist
    return maxx

def search_it(source, keyword):
    stem_kw = stem_it(keyword)
    stem_kw_pt = clean_pt(concat_tokens(stem_kw,postag_it(keyword)))
    if len(stem_kw_pt) == 0:
        return 0
    
    score = 0
    for skw in stem_kw_pt:
        ct = concat_tokens(stem_it(source),postag_it(source))
        if (skw[1] in [st[1] for st in ct]) or skw[0] in nltk.tokenize.word_tokenize(source):
            score += 1
        elif len(skw[0]) <= 1:
            score += 0
        else:
            score += get_total_dist(source,skw[0])*0.7
    return score/len(stem_kw_pt)

def general_search(sample, keyword):
    grades = []
    for sent in sample:
        grade = search_it(sent, keyword)
        grades.append([sent,grade])
    grades.sort(key = lambda x: x[1], reverse=True)
    grades = filter(lambda x: x[1] > 0.5, grades) 
    return [lg for lg in list(grades)]

def generate_detail(raw_list):
    regex = r"[0-9]+[ ][0-9][0-9.,]+"
    regex2 = r" pada [A-Z][^.,]+[.,]"
    bucket = ''
    res = []
    footer = 'IHPS I Tahun 2020'
    header = ['Permasalahan & Contohnya', 'Jumlah', 'Permasalahan', 'Nilai', '(Rp miliar)']

    for liner in raw_list[6:]:
        line = liner[0]
        if footer in line:
            continue
        if line in header:
            continue
        match = re.search(regex, line)  
        if match != None:
            entity = []
            nums = match.group(0)
            ket = line.replace(nums, '')
            if ket != '':
                entity.append(ket)
                match2 = re.findall(regex2, ket)
                if len(match2) == 0:
                    entity.append(getOrg(ket))
                else:
                    entity.append(clean_text(match2[0]))
                entity += nums.split(' ')
                res.append(entity)
            else:
                entity.append(bucket)
                match2 = re.findall(regex2, bucket)  
                if len(match2) == 0:
                    entity.append(getOrg(bucket))
                else:
                    entity.append(clean_text(match2[0]))
                entity += nums.split(' ')
                res.append(entity)
                bucket = ''
        else:
            bucket = (bucket + ' ' + line)

    for ress in res:
        ress[0] = re.sub(r'US\$[0-9.,]+ ', '', ress[0].strip().replace('\x17 ','').replace('â€¢ ',''))
    return res

raw_list = []

with open('tabel1_3.csv', 'r') as f:
    reader = csv.reader(f)
    raw_list = list(reader)
    
res = generate_detail(raw_list)
sample = [result[0] for result in res]

runn = True
while runn:
    keyword = input('Masukkan keyword: ')
    if keyword == 'exit':
        runn = False
    for gen in general_search(sample, keyword):
        print(gen)