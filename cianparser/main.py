'''файл для парсинга и скачивания данных с циан'''
import requests
from bs4 import BeautifulSoup
import re
import json
import os
import shutil

if __name__ == "__main__":
    session = requests.Session()
    session.headers = {
            'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.60 YaBrowser/20.12.0.963 Yowser/2.5 Safari/537.36',
                'Accept-Language': 'ru'
        }
    # для названия папок, индекс(число в названии) последней папки
    i=1
    # проходимся по страницам циан
    for k in range(1,10):

        url = f"https://cian.ru/cat.php?deal_type=rent&engine_version=2&offer_type=flat&p={str(k)}&region=2&room1=1&type=4&wp=1"
        res = session.get(url = url)
        res.raise_for_status()
        html = res.text
        soup = BeautifulSoup(html, 'lxml')
        offers = soup.select("div >article[data-name='CardComponent']")

        for block in offers:
            have_plan = False
            # получаем ссылку на объявление
            soup = BeautifulSoup(str(block),'lxml')
            oss = soup.select("div [data-name='LinkArea'] > a")
            result = re.findall(r'href=\S+[^"]', str(oss[0]))
            flat_url = result[0][6:-2]
            print(flat_url)    

            dir_path = os.path.join(r'./flat_data',f'flat_{str(i)}')
            os.makedirs(dir_path , exist_ok=True)
            with open(os.path.join(dir_path,'url.txt'),'w') as f:
                f.write(flat_url)
            
            # получаем перечень фото в изображении
            res = session.get(url = flat_url)            
            final = re.findall(r'"photos":\S+}],"phones"', str(res.text))[0][1:-2]
            final = final[9:-9].split('},')

            # если картинок меньше 8, то не скачиваем
            if len(final) <=15:
                print('Недостаточно изображений')
                shutil.rmtree(dir_path)
                pass
        
            elif 'isLayout' not in json.loads(final[0]+'}').keys():
                print('Нет плана квартиры')
                pass 
        
            else:
                for j in range(len(final)):
                    final[j]+='}'
                    img_info =  json.loads(final[j])
                    url = img_info['fullUrl']
                    img = requests.get(url = url, stream = True)

                    if img_info['isLayout']:
                        have_plan = True
                        path = os.path.join(dir_path ,'plan.jpg')
                    else:   
                        path = os.path.join(dir_path ,f'{j}.jpg')
                    if img.status_code == 200:
                        with open(path,'wb') as f:
                            shutil.copyfileobj(img.raw, f)
                

                print(have_plan)
                if not have_plan:
                    print('Нет плана')
                    shutil.rmtree(dir_path )
                else: i+=1
