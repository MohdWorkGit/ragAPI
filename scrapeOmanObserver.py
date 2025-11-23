import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.omanobserver.om/epaper"
DOMAIN = "https://www.omanobserver.om"
DOWNLOAD_DIR = "docs"
TRACK_FILE = "downloaded_pdfs.txt"

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Load already downloaded file list
if os.path.exists(TRACK_FILE):
    with open(TRACK_FILE, "r") as f:
        downloaded = set(line.strip() for line in f if line.strip())
else:
    downloaded = set()

# Fetch main page
response = requests.get(BASE_URL)
response.raise_for_status()
soup = BeautifulSoup(response.text, "html.parser")

new_files = 0

# Find all PDF links
for a in soup.find_all("a", href=True):
    if a["href"].lower().endswith(".pdf") or ".pdf?" in a["href"].lower():
        pdf_url = urljoin(DOMAIN, a["href"])
        filename = pdf_url.split("/")[-1].split("?")[0]

        if pdf_url not in downloaded:
            print(f"Downloading: {filename}")
            pdf_resp = requests.get(pdf_url, stream=True)
            pdf_resp.raise_for_status()

            file_path = os.path.join(DOWNLOAD_DIR, filename)
            with open(file_path, "wb") as f:
                for chunk in pdf_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            downloaded.add(pdf_url)
            new_files += 1

# Save updated download list
with open(TRACK_FILE, "w") as f:
    for url in sorted(downloaded):
        f.write(url + "\n")

if new_files == 0:
    print("No new PDFs found.")
else:
    print(f"Downloaded {new_files} new PDFs.")
