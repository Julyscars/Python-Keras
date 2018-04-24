from bs4 import BeautifulSoup
import requests
import lxml
import os ,re
header = {
        'User-Agent': "Mozilla/57.0 (Windows NT 8.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/62.0.1207.1 Safari/537.1"}
html = 'http://www.ximalaya.com/4083722/album/6495914/?page='
urls=[]
for page in range(1,3):
    url = html + str(page)
    urls.append(url)

for url in urls:
    start = requests.get(url, headers=header)
    soup = BeautifulSoup(start.text, 'lxml')
    link = soup.find_all('a', class_='title')[:-1]
    for a in link:
        i = 'www.ximalaya.com'
        href = a['href']
        name = a.get_text()
        b = href.split('/',)
        c,d,e =b[1],b[2],b[3]
        f='http://music.ifkdy.com/?url=http%3A%2F%2F' +i+'%2F'+c+'%2F'+d+'%2F'+e+'%2F'
        content = requests.get(f,headers=header)
        content_soup = BeautifulSoup(content.content,'lxml')
        links = content_soup.find_all('span',class_='am-input-group-btn')
        print(links)






"""

    for i in link:
        href = 'http://www.ximalaya.com/' + i['href']
        name = i.get_text()
        print(href, name)

"""




