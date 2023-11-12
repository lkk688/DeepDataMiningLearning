import urllib3

proxy = urllib3.ProxyManager("http://172.16.1.2:3128")#https does not work
proxy.request("GET", "https://google.com/")

import requests

proxies = {"http": "http://172.16.1.2:3128", "https": "http://172.16.1.2:3128"}
r = requests.get("https://www.google.com/", proxies=proxies, verify=False)
print(r)