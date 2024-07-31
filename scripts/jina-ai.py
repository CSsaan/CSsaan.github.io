import requests
from bs4 import BeautifulSoup

headerData = """
---
layout:     post
title:      "Hello ViTMatting"
subtitle:   " \"Hello World, Hello Blog\""
date:       2024-07-30 18:00:00
author:     "CS"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Meta
---

"""



url = 'https://r.jina.ai/https://blog.csdn.net/YangMax1/article/details/123241637'
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    print(headerData + str(soup))
else:
    print(f'Failed to retrieve content. Status code: {response.status_code}')

