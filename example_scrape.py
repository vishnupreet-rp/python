import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import csv

url = 'URL_HERE'

'''request website'''
soup = BeautifulSoup(requests.get(data).text, 'lxml')

links = []

'''extract all links under div tag'''
for divs in soup.find('div', {'class':'row'}):
    for link in soup.findAll('a'):
        links.append(link.get_text('href')


'''export all links to csv'''
with open('example.csv', 'w') as f:
     writer = csv.writer(f)
     writer.writerows(row for row in links if row)
