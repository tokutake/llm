"""Download a Japanese Wikipedia articles dump and extract clean text incrementally.

Uses only the Python standard library. The XML dump is streamed from bzip2 so
the compressed archive and full XML do not need to fit in memory.
"""
import argparse
import bz2
import html
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_URL = (
    "https://dumps.wikimedia.org/jawiki/latest/"
    "jawiki-latest-pages-articles-multistream.xml.bz2"
)
NS = "{http://www.mediawiki.org/xml/export-0.11/}"

def clean_wikitext(text):
    """Remove common markup/templates/tables; retain readable article prose."""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"<ref\b[^>]*/\s*>", " ", text, flags=re.I)
    text = re.sub(r"<ref\b[^>]*>.*?</ref\s*>", " ", text, flags=re.I|re.S)
    text = re.sub(r"<(math|gallery|nowiki|pre|code)\b[^>]*>.*?</\1\s*>", " ", text, flags=re.I|re.S)
    text = re.sub(r"\{\|.*?\|\}", " ", text, flags=re.S)
    # Nested templates: remove innermost groups repeatedly.
    for _ in range(10):
        updated = re.sub(r"\{\{[^{}]*\}\}", " ", text)
        if updated == text: break
        text = updated
    text = re.sub(r"\[\[(?:ファイル|File|画像|Image):[^\]]*\]\]", " ", text, flags=re.I)
    text = re.sub(r"\[\[([^\]|]+\|)?([^\]]+)\]\]", lambda m: m.group(2), text)
    text = re.sub(r"\[(?:https?://\S+\s+)?([^\]]+)\]", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"^\s*=+.*?=+\s*$", " ", text, flags=re.M)
    text = re.sub(r"^\s*[*#:;]+\s*", "", text, flags=re.M)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[\[\]{}|]", " ", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()

def open_dump(source):
    if source.startswith("http://") or source.startswith("https://"):
        req = urllib.request.Request(source, headers={"User-Agent": "TinyLLM-WikipediaPreparer/1.0 (local educational project)"})
        response = urllib.request.urlopen(req)
        if source.endswith(".bz2"):
            return bz2.BZ2File(response), response
        return response, response
    path = Path(source)
    f = open(path, "rb")
    if path.suffix == ".bz2": return bz2.BZ2File(f), f
    return f, f

def main():
    ap = argparse.ArgumentParser(description="Extract article text from a Japanese Wikipedia XML dump.")
    ap.add_argument("--source", default=DEFAULT_URL, help="Dump URL or local XML/XML.bz2 path")
    ap.add_argument("--out", default="wikipedia_ja.txt", help="Output UTF-8 corpus file")
    ap.add_argument("--max-mb", type=float, default=20, help="Stop after writing this many MiB (0 means unlimited)")
    ap.add_argument("--min-chars", type=int, default=200, help="Skip shorter article bodies")
    ap.add_argument("--every", type=int, default=10000, help="Progress message interval in pages")
    args = ap.parse_args()
    stream, owner = open_dump(args.source)
    written = pages = kept = 0
    limit = int(args.max_mb * 1024 * 1024) if args.max_mb > 0 else None
    try:
        with open(args.out, "w", encoding="utf-8") as out:
            for event, elem in ET.iterparse(stream, events=("end",)):
                if elem.tag != NS + "page": continue
                pages += 1
                ns = elem.findtext(NS + "ns", default="")
                redirect = elem.find(NS + "redirect")
                title = elem.findtext(NS + "title", default="")
                revision = elem.find(NS + "revision")
                text_node = revision.find(NS + "text") if revision is not None else None
                raw = text_node.text or "" if text_node is not None else ""
                # Keep main-namespace articles only; skip redirects and likely disambiguation pages.
                if ns == "0" and redirect is None and "曖昧さ回避" not in raw:
                    body = clean_wikitext(raw)
                    if len(body) >= args.min_chars:
                        record = f"\n\n### {title}\n{body}\n"
                        encoded_len = len(record.encode("utf-8"))
                        if limit is not None and written + encoded_len > limit: break
                        out.write(record); written += encoded_len; kept += 1
                elem.clear()
                if args.every and pages % args.every == 0:
                    print(f"pages={pages:,} articles={kept:,} written={written/1048576:.1f} MiB", file=sys.stderr, flush=True)
    finally:
        stream.close(); owner.close()
    print(f"Done: {kept:,} articles, {written/1048576:.2f} MiB -> {args.out}")

if __name__ == "__main__": main()
