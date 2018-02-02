# -*- coding: UTF-8 -*-
import requests
from bs4 import BeautifulSoup
class mafengwo():
    def __init__(self):
        self.headers = {'User-Agent': "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"}

    def all_url(self,url):
        html = requests.get(url)
        soup = BeautifulSoup(html.content, 'html.parser').find('div',class_='postList').find_all('h2')
        n = []
        for i in soup:
            nr = i.find('a')
            title = nr.get_text()
            n.append(title)
            self.save(n)

    def save(self,n):
        with open(r'travl1.txt', 'w', encoding='utf-8') as f:
            for j in n:
                f.write(j)
                f.write('\n')
            f.close()
    def request(self, url):
        content = requests.get(url, headers=self.headers)
        return content

Mafengwo = mafengwo()
Mafengwo.all_url('http://360.mafengwo.cn/travels/index.php?f=0&p=1')
print('saving')