#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tamil Nadu Nursing Council (SPA/hash routes) text scraper.
- Renders JS with Selenium (headless Chrome)
- Crawls internal links including #/... routes
- Saves all page text into a single TXT file
- Logs visited URLs

Usage (examples):
  python tnnc_spa_text_scraper.py --start https://tamilnadunursingcouncil.com/ --max-pages 300 --out text_out.txt
  python tnnc_spa_text_scraper.py --headless false --delay 2

Requires:
  pip install selenium webdriver-manager beautifulsoup4
"""

import argparse
import time
import re
from collections import deque
from urllib.parse import urlparse, urljoin, urldefrag

from bs4 import BeautifulSoup
from urllib import robotparser

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def clean_url(u: str, keep_fragment: bool = True) -> str:
    if not u:
        return ""
    u = u.strip()
    if keep_fragment:
        # normalize whitespace and strip
        return u
    # drop fragment when checking robots/host
    return urldefrag(u)[0]


def is_same_origin(u: str, origin: str) -> bool:
    try:
        return urlparse(u).netloc == urlparse(origin).netloc
    except Exception:
        return False


def make_browser(headless: bool) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1366,900")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--user-agent=Mozilla/5.0 (compatible; TNNC-SPA-Scraper/1.0)")
    driver = webdriver.Chrome(service=webdriver.chrome.service.Service(ChromeDriverManager().install()), options=opts)
    return driver


def wait_for_render(driver, wait_secs: float):
    # 1) wait DOM ready
    WebDriverWait(driver, max(1, int(wait_secs))).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )
    # 2) small extra delay lets SPA routers/renderers finish
    time.sleep(max(0.5, wait_secs / 2.0))


def lazy_scroll(driver, steps: int = 4, pause: float = 0.4):
    # Try to trigger lazy content
    last_h = 0
    for _ in range(steps):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(pause)
        new_h = driver.execute_script("return document.body.scrollHeight || 0;")
        if new_h == last_h:
            break
        last_h = new_h


def extract_visible_text_from_dom(driver) -> str:
    # Get current DOM HTML and strip non-content nodes
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # removing obvious chrome like nav/footer/forms
    for tag in soup.find_all(["nav", "footer", "form", "aside"]):
        # keep if you think important; we remove to keep text clean
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def collect_links(driver, base_origin: str):
    """Collect internal links including #/ routes."""
    links = set()

    # All <a> hrefs
    for a in driver.find_elements(By.TAG_NAME, "a"):
        href = (a.get_attribute("href") or "").strip()
        if not href:
            continue

        # Normalize: turn relative or hash into absolute:
        # - If href starts with "#/" it’s an SPA route on same page
        if href.startswith("#/"):
            href = urljoin(base_origin, href)  # origin + '#/route'

        # Some routers inject same-origin absolute already; some links might be javascript:void(0)
        if href.lower().startswith("javascript:"):
            continue

        # Keep same-origin only
        if is_same_origin(href, base_origin):
            links.add(href)

    return links


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="https://tamilnadunursingcouncil.com/", help="Start URL")
    ap.add_argument("--max-pages", type=int, default=200, help="Maximum pages to visit")
    ap.add_argument("--delay", type=float, default=1.5, help="Polite delay per page (seconds)")
    ap.add_argument("--headless", default="true", choices=["true", "false"], help="Run headless")
    ap.add_argument("--out", default="tamilnadu_nursing_council.txt", help="Output TXT file")
    ap.add_argument("--log-urls", default="urls_visited.txt", help="Visited URLs log")
    args = ap.parse_args()

    start_url = args.start.strip()
    base_origin = f"{urlparse(start_url).scheme}://{urlparse(start_url).netloc}"
    same_host_only = urlparse(start_url).netloc

    # robots.txt (note: fragments are ignored by servers; check robots against URL without fragment)
    rp = robotparser.RobotFileParser()
    rp.set_url(urljoin(base_origin, "/robots.txt"))
    try:
        rp.read()
    except Exception:
        pass

    driver = make_browser(headless=(args.headless.lower() == "true"))

    visited = set()
    q = deque([start_url])
    saved = 0

    with open(args.out, "w", encoding="utf-8") as outf, open(args.log_urls, "w", encoding="utf-8") as logf:
        try:
            while q and (args.max_pages <= 0 or saved < args.max_pages):
                url = q.popleft()
                if url in visited:
                    continue
                # Only same host
                if urlparse(url).netloc != same_host_only:
                    continue

                # robots check on URL without fragment
                robots_check_url = clean_url(url, keep_fragment=False)
                can_fetch = True
                try:
                    can_fetch = rp.can_fetch("*", robots_check_url)
                except Exception:
                    pass
                if not can_fetch:
                    continue

                try:
                    driver.get(url)
                    wait_for_render(driver, args.delay)
                    lazy_scroll(driver)
                except Exception:
                    continue

                # Extract text
                try:
                    text = extract_visible_text_from_dom(driver)
                except Exception:
                    text = ""

                # Write section
                divider = "=" * 90
                outf.write(f"\n\n{divider}\nPAGE: {url}\n{divider}\n{text}\n")
                outf.flush()

                logf.write(url + "\n")
                logf.flush()

                visited.add(url)
                saved += 1
                print(f"[{saved}] saved: {url}")

                # Collect and enqueue new links (including #/ routes)
                try:
                    links = collect_links(driver, base_origin)
                except Exception:
                    links = set()

                for link in links:
                    # Keep hash-variants (SPA pages) distinct to capture their rendered content
                    if link not in visited:
                        # Normalize a bit: ensure we keep hash routes as-is
                        link = clean_url(link, keep_fragment=True)
                        if is_same_origin(link, base_origin):
                            q.append(link)

        finally:
            driver.quit()

    print(f"\n✅ done. extracted text saved to: {args.out}\n📝 urls visited logged in: {args.log_urls}")


if __name__ == "__main__":
    main()
