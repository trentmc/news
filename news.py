#!/usr/bin/env python3
"""Rewrite news articles as 5 cynical bullet points using Claude."""

import anthropic
import httpx
from readability import Document
from bs4 import BeautifulSoup

SYSTEM_PROMPT = """You are a cynical news analyst. When given a news article, rewrite it as exactly 5 bullet points.

For each bullet point, reveal the likely real motivations of the subjects involved. Assume every actor is driven by some combination of:
- Money
- Power
- Sex
- Ego / Vanity
- Fear
- Tribal loyalty
- Legacy

Be concise, sharp, and darkly honest. Name the motivation explicitly in each bullet."""

client = anthropic.Anthropic()


def fetch_article(url: str) -> str:
    """Fetch a URL and extract the article text."""
    resp = httpx.get(url, follow_redirects=True, timeout=15,
                     headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    doc = Document(resp.text)
    soup = BeautifulSoup(doc.summary(), "html.parser")
    return f"Title: {doc.title()}\n\n{soup.get_text(separator='\n', strip=True)}"


def rewrite(article_text: str) -> str:
    """Send the article to Claude and get 5 bullet points back."""
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": article_text}],
    )
    return msg.content[0].text


def main():
    print("=== Cynical News Rewriter ===")
    print("Enter a news article URL (or 'q' to quit).\n")
    while True:
        url = input("URL> ").strip()
        if not url or url.lower() == "q":
            break
        try:
            print("\nFetching article...")
            article = fetch_article(url)
            print("Analyzing with Claude...\n")
            bullets = rewrite(article)
            print(bullets)
        except httpx.HTTPStatusError as e:
            print(f"HTTP error fetching article: {e}")
        except Exception as e:
            print(f"Error: {e}")
        print()


if __name__ == "__main__":
    main()
