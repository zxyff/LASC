import pandas as pd
import csv
import time
import requests
import csv
from DrissionPage import ChromiumPage
from lxml import etree
page = ChromiumPage(9876)

df = pd.read_excel('address-ENS.xlsx')
for i in df.values:
    ym = i[1]
    url = f'https://app.ens.domains/{ym}?tab=ownership'
    page.get(url)
    time.sleep(4)
    html = etree.HTML(page.html)
    trs = html.xpath('//div[@class="sc-a073d725-0 fJDnAW"]')
    for tr in trs:
        uids = ''.join(tr.xpath('./@data-testid'))
        owner = ''.join(tr.xpath('.//button[@data-testid="role-tag-owner"]/div/text()'))
        manger = ''.join(tr.xpath('.//button[@data-testid="role-tag-manager"]/div/text()'))
        with open('数据.csv', 'a+', encoding='utf-8-sig', newline='') as f:
            f = csv.writer(f)
            f.writerow(list(i) + [uids, owner, manger])