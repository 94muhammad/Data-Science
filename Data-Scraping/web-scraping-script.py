#!/usr/bin/env python
# coding: utf-8

# # Web-scraping

# ### Website: Fat llama

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
import requests
import json


# #### Create function to iterate through page links

# In[2]:


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

    for link in sitemap_list[20:30]:
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


# In[3]:


# testing timeit()
 
import timeit
import random

url = "https://fatllama.com/sitemap_index.xml"



starttime = timeit.default_timer()
print("The start time is :",starttime)
scrap(url)

print("The time difference is :", timeit.default_timer() - starttime)


# In[4]:


scrap.all_addresses[:10]


# In[5]:


Name = []
Price = []
mainCategory = []
subCategory = []
subCategoryII = []

val = []

for i in scrap.all_addresses[:10000]:

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


# In[6]:


df = pd.DataFrame(list(zip(Name, mainCategory, subCategory, subCategoryII, Price)), columns = ['Product_name', 'Category', 'Sub_Category', 'Sub_CategoryII', 'Rental Price'])


# In[7]:


df


# In[8]:


df.to_csv('0_to_1k_items.csv')


# In[ ]:





# ## testing approach for function above

# In[ ]:


test = requests.get('https://fatllama.com/sitemap_index.xml').text
test_links = BeautifulSoup(test,'lxml')
#print(test_links.prettify())
sitemap_url = test_links.find_all('sitemap')

sitemap_list = []

for i in sitemap_url:
    sitemap_links = i.find('loc').text
    sitemap_list.append(sitemap_links)


# In[ ]:


sitemap_url_list = []

for link in sitemap_list[2:]:
    fetch_url = requests.get(link).text
    next_sitemap = BeautifulSoup(fetch_url,'lxml')
    urls = next_sitemap.find_all('url')
    
    sitemap_url_list.append(urls)


# In[ ]:


len(sitemap_url_list[10])


# In[ ]:


a = []

for i in range(len(sitemap_url_list)):
    for j in sitemap_url_list[i]:
        
        loc = j.find('loc').text
    
        a.append(loc)


# In[ ]:


print(a[:10])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #### Using Sitemaps to automate data scraping from pages

# In[ ]:


sitemap_1 = requests.get("https://fatllama.com/2/sitemap.xml").text
print(sitemap_1)


# #### Using BeautifulSoup to get HTML/Json 

# In[ ]:


links_to_scrap = BeautifulSoup(sitemap_1, 'lxml')
print(links_to_scrap.prettify())


# #### Narrow down the search 

# In[ ]:


get_links = links_to_scrap.find_all('url')
print(get_links)


# #### Push web addresses into a list

# In[ ]:


links = []
for i in get_links:
    
    pull_links_from_html = i.find('loc').text
    links.append(pull_links_from_html)

print(links)


# In[ ]:


for i in links:
    print(i)


# #### Iterate through web-address list to extract Name and Price of relevant item

# In[ ]:


Name = []
Price = []
mainCategory = []
subCategory = []
subCategoryII = []

val = []

for i in links:

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
        itemCategory = data.find_all('p', cursor='pointer')[2].text
    
    except:
        
        pass
    
    Name.append(itemNames)
    Price.append(price)
    mainCategory.append(mainCat)
    subCategory.append(subCat)
    subCategoryII.append(itemCategory)


# In[ ]:


#name = []
#price = []
#mainCategory = []
#subCategory = []

#for i in links:
#    fetch_html_text = requests.get(i).text
#    data = BeautifulSoup(fetch_html_text, 'lxml')
#    itemNames = data.find('h1').text
#    itemPrices = data.find('span', class_='sc-crzoUp Sxavm')
#    mainCat = data.find('p', cursor='pointer')[0]
#    subCat = data.find('p', cursor='pointer')[1]
    
#    name.append(itemNames)
#    price.append(itemPrices)
#    mainCategory.append(mainCat)
#    subCategory.append(subCat)


# In[ ]:


df = pd.DataFrame(list(zip(Name, mainCategory, subCategory, subCategoryII, Price)), columns = ['Product_name', 'Category', 'Sub_Category', 'Sub_CategoryII', 'Rental Price'])


# In[ ]:


df


# In[ ]:


df['Rental Price'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')


# In[ ]:


df


# In[ ]:


df.to_csv('sitemap2.csv')


# In[ ]:


dslr_cam = df.loc[df['Sub_CategoryII'].str.contains("DSLR Cameras", case=False)]


# In[ ]:


dslr_cam


# In[ ]:


(sum(dslr_cam['Rental Price'].astype(str).astype(int)))/(len(dslr_cam['Rental Price']))


# In[ ]:





# In[ ]:





# In[ ]:




