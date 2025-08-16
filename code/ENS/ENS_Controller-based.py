import time
import requests
import os
import re
import csv
import pandas as pd
from lxml import etree
from DrissionPage._pages.chromium_page import ChromiumPage
from loguru import logger
page = ChromiumPage()

df = pd.read_excel('ens-common_data.xlsx')
for i in df.values:
    uid = i[0]
    url = f'https://app.ens.domains/{uid}'
    page.get(url)
    for num in range(5):
        page.scroll.to_bottom()
        time.sleep(0.5)
    html = etree.HTML(page.html)
    trs = html.xpath('//div[@data-testid="names-list"]/div/div/a')
    if trs == []:
        link = '0'
        with open('信息.csv', 'a+', encoding='utf-8-sig', newline='') as fi:
            fi = csv.writer(fi)
            fi.writerow(
                list(i) + [link]
            )
    else:
        link = 'X'
        for tr in trs:
            glz = tr.xpath('.//div[@data-testid="tag-name.manager-true"]/text()')
            syz = tr.xpath('.//div[@data-testid="tag-name.owner-true"]/text()')
            if glz == [] or syz == []:
                link = ''.join(tr.xpath('.//div[@class="sc-72ba6205-0 iOENaH"]//text()'))
                with open('信息.csv', 'a+', encoding='utf-8-sig', newline='') as fi:
                    fi = csv.writer(fi)
                    fi.writerow(
                        list(i) + [link]
                    )
        if link == 'X':
            with open('信息.csv', 'a+', encoding='utf-8-sig', newline='') as fi:
                fi = csv.writer(fi)
                fi.writerow(
                    list(i) + [link]
                )