## news

Quickstart
```
git clone https://github.com/trentmc/news.git

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

python news.py
```

## Evolving the prompt

`evolve.py` runs a genetic algorithm over the prompt in `news.py`.
Claude acts as the variation operator (mutation + crossover) and as the fitness
judge, scoring each candidate's 5-bullet output on cynicism, insight,
concision, and format. The `BASE_SYSTEM_PROMPT` seeds generation 0; the best
prompt found is written to `evolved_prompt.txt`.

```
python evolve.py                      # offline, built-in sample articles
python evolve.py --pop 8 --gens 5     # bigger search
python evolve.py --urls urls.txt      # evolve against a real article corpus
```

Cost is roughly `pop * eval_size * 2` Claude calls per generation, so keep the
defaults small while experimenting. Run `python evolve.py --help` for all knobs.

`news.py` automatically uses `evolved_prompt.txt` when it exists, falling back
to the built-in `BASE_SYSTEM_PROMPT` otherwise. To override:

- `NEWS_USE_BASE_PROMPT=1 python news.py` — force the baseline prompt
- `NEWS_PROMPT_FILE=path/to/prompt.txt python news.py` — use a specific prompt