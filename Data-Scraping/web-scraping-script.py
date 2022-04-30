# # Web-scraping

# ### Website: Fat llama

# ### Import Libraries


import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
import requests
import json


# #### Create function to iterate through page links

def scrap(url):
    
    data = requests.get(url).text
    sitemaps = BeautifulSoup(data,'lxml')
    
    sitemap_url = sitemaps.find_all('sitemap')

    sitemap_list = []
    
    # loop to fetch sitemap url and create a list
    for i in sitemap_url:
        sitemap_links = i.find('loc').text
        sitemap_list.append(sitemap_links)
     
    # Second loop to fetch data from each link present in the sitemap list created in the loop above
    sitemap_url_list = []

    for link in sitemap_list:
        fetch_url = requests.get(link).text
        next_sitemap = BeautifulSoup(fetch_url,'lxml')
        urls = next_sitemap.find_all('url')
        
        sitemap_url_list.append(urls)
    
    # get item web - addresses from the sitemaps
    
    scrap.all_addresses = []
    for i in range(len(sitemap_url_list)):
        for j in sitemap_url_list[i]:
        
            loc = j.find('loc').text
    
            scrap.all_addresses.append(loc)
        
    print(len(scrap.all_addresses))



# testing timeit()
 
import timeit
import random

url = "https://fatllama.com/sitemap_index.xml"



starttime = timeit.default_timer()
print("The start time is :",starttime)
scrap(url)

print("The time difference is :", timeit.default_timer() - starttime)


# get item name, categories, and rental price for each item from its webpage

Name = []
Price = []
mainCategory = []
subCategory = []
subCategoryII = []

val = []

for i in scrap.all_addresses:

    fetch_html_text = requests.get(i).text
    data = BeautifulSoup(fetch_html_text, 'lxml')

    try:
        
        itemPrices = data.find_all('span')

        for i in range(len(itemPrices)): # iterate through number of span tags 
            a = itemPrices[i].text
            for character in a:
                if character.isdigit(): # check if span tag has a digit
                    val=i
                    price = itemPrices[val].text

    except:
        pass
    
       
    try:
        itemNames = data.find('h1').text
        mainCat = data.find_all('p', cursor='pointer')[0].text
        subCat = data.find_all('p', cursor='pointer')[1].text
        itemCat = data.find_all('p', cursor='pointer')[2].text
    
    except:
        
        pass
    
    Name.append(itemNames)
    Price.append(price)
    mainCategory.append(mainCat)
    subCategory.append(subCat)
    subCategoryII.append(itemCat)

# create a dataframe 
df = pd.DataFrame(list(zip(Name, mainCategory, subCategory, subCategoryII, Price)), columns = ['Product_name', 'Category', 'Sub_Category', 'Sub_CategoryII', 'Rental Price'])



