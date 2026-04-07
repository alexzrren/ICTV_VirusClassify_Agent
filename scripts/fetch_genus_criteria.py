#!/usr/bin/env python3
"""
Fetch ICTV genus-level report pages and save raw HTML content for parsing.
URLs: https://ictv.global/report/chapter/{family}/{family}/{genus} (all lowercase)

Saves each page to data/genus_reports/{Family}/{Genus}.txt
"""

import os
import sys
import time
import sqlite3
import json
import re
import urllib.request
import urllib.error
from pathlib import Path
from html.parser import HTMLParser


class HTMLToText(HTMLParser):
    """Simple HTML to text converter."""
    def __init__(self):
        super().__init__()
        self._text = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'nav', 'footer', 'header'):
            self._skip = True
        if tag in ('p', 'br', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'tr'):
            self._text.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'nav', 'footer', 'header'):
            self._skip = False
        if tag in ('p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'table'):
            self._text.append('\n')

    def handle_data(self, data):
        if not self._skip:
            self._text.append(data)

    def get_text(self):
        return ''.join(self._text)


def html_to_text(html: str) -> str:
    parser = HTMLToText()
    parser.feed(html)
    return parser.get_text()


def fetch_url(url: str, retries: int = 2) -> str | None:
    """Fetch URL content, return text or None on failure."""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; ICTVBot/1.0)',
                'Accept': 'text/html',
            })
            # Clear proxy env vars
            old_proxies = {}
            for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
                if k in os.environ:
                    old_proxies[k] = os.environ.pop(k)

            handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(handler)

            with opener.open(req, timeout=30) as resp:
                data = resp.read().decode('utf-8', errors='replace')

            # Restore proxy env vars
            os.environ.update(old_proxies)
            return data
        except Exception as e:
            # Restore proxy env vars
            for k, v in old_proxies.items():
                os.environ[k] = v
            if attempt < retries:
                time.sleep(2)
            else:
                print(f"  FAILED: {e}", file=sys.stderr)
                return None


def get_genera_to_fetch(db_path: str, target_families: dict) -> list[tuple[str, str]]:
    """Return list of (family, genus) to fetch."""
    db = sqlite3.connect(db_path)
    result = []
    for fam, genera in sorted(target_families.items()):
        if genera is None:
            rows = db.execute(
                "SELECT DISTINCT genus FROM species WHERE family=? AND genus IS NOT NULL ORDER BY genus",
                (fam,)
            ).fetchall()
            genera = [r[0] for r in rows]
        for g in genera:
            result.append((fam, g))
    db.close()
    return result


def main():
    db_path = "data/taxonomy.db"
    outdir = Path("data/genus_reports")

    # Families needing genus-level criteria
    target_families = {
        "Amnoonviridae": None,
        "Asfarviridae": None,
        "Astroviridae": None,
        "Pneumoviridae": None,
        "Bornaviridae": None,
        "Flaviviridae": None,
        "Hepeviridae": None,
        "Arenaviridae": None,
        "Adenoviridae": None,
        "Sedoreoviridae": None,
        "Peribunyaviridae": None,
        "Filoviridae": None,
        "Spinareoviridae": None,
        "Caliciviridae": None,
        "Picobirnaviridae": None,
        "Paramyxoviridae": ["Respirovirus", "Morbillivirus", "Henipavirus", "Orthorubulavirus", "Metaavulavirus"],
        "Orthoherpesviridae": ["Simplexvirus", "Varicellovirus", "Cytomegalovirus", "Roseolovirus", "Lymphocryptovirus", "Rhadinovirus"],
        "Parvoviridae": ["Protoparvovirus", "Bocaparvovirus", "Dependoparvovirus", "Erythroparvovirus", "Chaphamaparvovirus"],
        "Poxviridae": ["Orthopoxvirus", "Parapoxvirus", "Avipoxvirus", "Molluscipoxvirus", "Leporipoxvirus"],
        "Rhabdoviridae": ["Lyssavirus", "Vesiculovirus", "Ephemerovirus", "Novirhabdovirus", "Ledantevirus", "Sigmavirus"],
        "Picornaviridae": ["Enterovirus", "Hepatovirus", "Parechovirus", "Kobuvirus", "Aphthovirus", "Cardiovirus", "Senecavirus", "Sapelovirus"],
    }

    pairs = get_genera_to_fetch(db_path, target_families)
    print(f"Total genera to fetch: {len(pairs)}")

    ok = 0
    fail = 0
    skip = 0

    for i, (fam, genus) in enumerate(pairs):
        out_path = outdir / fam / f"{genus}.txt"
        if out_path.exists() and out_path.stat().st_size > 500:
            print(f"[{i+1}/{len(pairs)}] SKIP {fam}/{genus} (exists)")
            skip += 1
            continue

        url = f"https://ictv.global/report/chapter/{fam.lower()}/{fam.lower()}/{genus.lower()}"
        print(f"[{i+1}/{len(pairs)}] {fam}/{genus} ...", end=" ", flush=True)

        html = fetch_url(url)
        if html is None:
            fail += 1
            continue

        text = html_to_text(html)

        # Save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding='utf-8')
        size = len(text)
        print(f"OK ({size} chars)")
        ok += 1

        time.sleep(1)  # rate limit

    print(f"\nDone: {ok} fetched, {skip} skipped, {fail} failed")


if __name__ == "__main__":
    main()
