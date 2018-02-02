# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 11:04:26 2017

@author: Atlantis
"""

import os
import re
from tqdm import tqdm
from html.parser import HTMLParser




patternStr = r'%s(.+?)%s'%('http://', '.sohu')
pattern = re.compile(patternStr)

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
#        print("Start tag:", tag)
        #定位栏目
        self.starttag = tag
    def handle_endtag(self, tag):
#        print("End tag  :", tag)
        self.endtag = tag
        if self.endtag == 'doc':
            self.txt.close()
    def handle_data(self, data):
        if data.strip():
            if self.starttag == 'url':
                url = data
                m = re.match(pattern, url)
                typ = m.group(1)
                self.col = os.path.join('/home/dl/NLP', 'sogou2', typ)
                if not os.path.exists(self.col):
                    os.makedirs(self.col)
            #创建文本文件
            if self.starttag == 'docno':
                self.txt = open(os.path.join(self.col, data + '.txt'), 'w', encoding='gb18030')
            if self.starttag == 'content':
                self.txt.write(data)
parser = MyHTMLParser()

sogou = '/home/dl/NLP/SogouCS.reduced'
for f in tqdm(os.listdir(sogou)):
    txt = open(os.path.join(sogou, f), encoding='gb18030')
    text = txt.read()
    parser.feed(text)