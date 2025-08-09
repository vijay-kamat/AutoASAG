import requests
from bs4 import BeautifulSoup
from typing import List

def fetch_text_from_url(url: str) -> str:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = " ".join(p.get_text(strip=True) for p in soup.find_all("p"))
    return text

def fetch_multiple(urls: List[str]) -> dict:
    out = {}
    for u in urls:
        try:
            out[u] = fetch_text_from_url(u)
        except Exception as e:
            out[u] = None
    return out
