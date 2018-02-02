from bs4 import BeautifulSoup
import requests


if __name__ == "__main__":
    #创建txt文件
    file = open('wsctq.txt', 'w', encoding='utf-8')
    #目录地址
    target_url = 'http://www.quanxue.cn/CT_NanHuaiJin/CanTongIndex.html'
    #User-Agent
    head = {}
    head['User-Agent'] = 'Mozilla/5.0 (Linux; Android 4.1.1; Nexus 7 Build/JRO03D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166  Safari/535.19'
    target_html = requests.get(url = target_url, headers = head)
    #创建BeautifulSoup对象
    listmain_soup = BeautifulSoup(target_html.content,'lxml')
    #搜索文档树,找出div标签中class为index_left_td的所有子标签
    book_name = listmain_soup.find_all('a',class_='index_left_td')
    for i in book_name:
        href = i.find('a')['href']
        name = i.get_text()[2:]
        html_ture = 'http://www.quanxue.cn/CT_NanHuaiJin/'+ href
        txt_url = requests.get(html_ture,headers=head)
        txt_soup = BeautifulSoup(txt_url.content,'lxml')
        txt_content = txt_soup.find('div',class_='main').find_all('p')
        file.write(format(name,"^")+'\n')
        for p in txt_content:
            text = p.get_text()
            text = text.strip()
            file.write( str(text)+'\n')


    print("sved")
    file.close()



