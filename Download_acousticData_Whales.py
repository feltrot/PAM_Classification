

# download audio files from Watkins Marine Mammal Sound Database (and others)


import requests
from bs4 import BeautifulSoup
import os
import ssl
import urllib3


class CustomHttpAdapter (requests.adapters.HTTPAdapter):
    '''Transport adapter" that allows us to use custom ssl_context.'''

    def __init__(self, ssl_context=None, **kwargs):
        self.ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections, maxsize=maxsize,
            block=block, ssl_context=self.ssl_context)

ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
ctx.options |= 0x4
session.mount('https://', CustomHttpAdapter(ctx))



import ssl
import urllib.request


def download_files(url, local_folder):
    # Set up SSL context to allow legacy TLS versions
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT

    # Send a GET request to the URL
    response = urllib.request.urlopen(webpage_url, context=ctx)

    #response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all the links on the page
        links = soup.find_all('a', href=True)

        # Create the local folder if it doesn't exist
        os.makedirs(local_folder, exist_ok=True)

        # Loop through each link and download the file
        for link in links:
            file_url = link['href']

            # Download the file
            download_file(file_url, local_folder)

    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

def download_file(file_url, local_folder):
    # Send a GET request to the file URL
    response = requests.get(file_url, verify=False)

    # Extract the file name from the URL
    file_name = os.path.join(local_folder, file_url.split("/")[-1])

    # Save the file locally
    with open(file_name, 'wb') as file:
        file.write(response.content)
        print(f"Downloaded: {file_name}")

# Example usage
webpage_url = "https://whoicf2.whoi.edu/science/B/whalesounds/bestOf.cfm?code=BE7A"
local_folder_path = "Documents/EMEC/Acoustics/Data_Acoustics/SpeciesCalls/Orca/downloaded_files"
#local_folder_path = "./downloaded_files"

download_files(webpage_url, local_folder_path)  # folder path is created if not already existing












import ssl
import warnings

import requests
import requests.packages.urllib3.exceptions as urllib3_exceptions

warnings.simplefilter("ignore", urllib3_exceptions.InsecureRequestWarning)


class TLSAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")
        ctx.options |= 0x4
        kwargs["ssl_context"] = ctx
        return super(TLSAdapter, self).init_poolmanager(*args, **kwargs)

url = "http://naijasyncor.huma-num.fr/carte/mp3/LAG_01_Hairdressing_M.wav"

with requests.session() as s:
    s.mount("https://", TLSAdapter())

    response = s.get(url)
    with open("LAG_01_Hairdressing_M.wav", "wb") as f_out:
        f_out.write(response.content)





with requests.session() as s:
    s.mount("https://", TLSAdapter())

    response = s.get(webpage_url)
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')
    # Find all the links on the page
    links = soup.find_all('a', href=True)

    # Create the local folder if it doesn't exist
    os.makedirs(local_folder_path, exist_ok=True)
    
    # Loop through each link and download the file
    for link in links:
        file_url = link['href']

        with open(os.path.join(local_folder_path, "Orca_Calls_01.wav"), "wb") as f_out:
            f_out.write(response.content)

    #else:
    #    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")





# find all the URLs in one webpage
#---------------------------------
import requests
from bs4 import BeautifulSoup
 
 
webpage_url = "https://whoicf2.whoi.edu/science/B/whalesounds/bestOf.cfm?code=BE7A"
reqs = requests.get(webpage_url)
soup = BeautifulSoup(reqs.text, 'html.parser')
links = soup.find_all('a')

with requests.session() as s:
    s.mount("https://", TLSAdapter())
    
    urls = []
    for link in links:
        print(link.get('href'))





import time
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

# URLs
urls = ['https://www.northwestknowledge.net/metdata/data/pr_1979.nc', 'https://www.northwestknowledge.net/metdata/data/pr_1980.nc', 'https://www.northwestknowledge.net/metdata/data/pr_1981.nc', 'https://www.northwestknowledge.net/metdata/data/pr_1982.nc']