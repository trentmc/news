#!/usr/bin/env python3
"""Rewrite news articles as 5 cynical bullet points using Claude."""

import os
import sys
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv
from readability import Document
from bs4 import BeautifulSoup

load_dotenv()

# Baseline prompt. This also seeds generation 0 of evolve.py's prompt GA.
BASE_SYSTEM_PROMPT = """You are a cynical news analyst. When given a news article, rewrite it as exactly 5 bullet points.

For each bullet point, reveal the likely real motivations of the subjects involved. Assume every actor is driven by some combination of:
- Money
- Power
- Sex
- Ego / Vanity
- Fear
- Tribal loyalty
- Legacy

Be concise, sharp, and darkly honest. Name the motivation explicitly in each bullet."""

# If evolve.py has produced a better prompt, prefer it; otherwise use the
# baseline. Set NEWS_PROMPT_FILE to point at a different prompt, or
# NEWS_USE_BASE_PROMPT=1 to force the baseline.
_PROMPT_FILE = Path(os.environ.get(
    "NEWS_PROMPT_FILE", Path(__file__).with_name("evolved_prompt.txt")))


def load_system_prompt() -> str:
    """Return the evolved prompt if present, else the baseline."""
    if os.environ.get("NEWS_USE_BASE_PROMPT"):
        return BASE_SYSTEM_PROMPT
    try:
        text = _PROMPT_FILE.read_text().strip()
        if text:
            return text
    except OSError:
        pass
    return BASE_SYSTEM_PROMPT


SYSTEM_PROMPT = load_system_prompt()

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit(
        "ANTHROPIC_API_KEY is not set.\n"
        "Add it to a .env file in this directory (see .env.example):\n"
        "    ANTHROPIC_API_KEY=sk-ant-...\n"
        "or export it in your shell before running."
    )

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
