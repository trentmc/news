#!/usr/bin/env python3
"""Evolve the cynical-news SYSTEM_PROMPT with a genetic algorithm.

Claude plays three roles here:
  * the model under test   - runs a candidate prompt on real articles
  * the variation operator - mutates and crosses over prompts
  * the fitness judge      - scores each candidate's output

The current SYSTEM_PROMPT from news.py seeds generation 0. Each generation we
evaluate every candidate against a fixed sample of articles, keep the elites,
and breed the rest via tournament selection + crossover + mutation. The best
prompt found is written to evolved_prompt.txt.

Usage:
    python evolve.py                      # offline, built-in sample articles
    python evolve.py --urls urls.txt      # fetch + cache a real corpus
    python evolve.py --pop 8 --gens 5 --eval-size 3

Costs scale as roughly  pop * eval_size * 2  Claude calls per generation
(one rewrite + one judge per candidate/article), plus a handful for breeding.
Keep the defaults small while experimenting.
"""

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Importing news runs its ANTHROPIC_API_KEY check and builds the shared client.
# Seed evolution from the BASE prompt, not news.SYSTEM_PROMPT — the latter may
# already be a prior evolved_prompt.txt, which would make runs compound on
# themselves and break --seed reproducibility.
from news import client, fetch_article, BASE_SYSTEM_PROMPT as SYSTEM_PROMPT

# --- models -----------------------------------------------------------------
# The candidate prompts are optimised for the production model, so the rewrite
# under test uses the same model news.py ships with. Mutation and judging are
# cheaper meta-tasks, so they default to Haiku.
GEN_MODEL = "claude-sonnet-4-6"   # runs the candidate prompt (matches news.py)
META_MODEL = "claude-haiku-4-5"   # mutate / crossover / judge (cheap meta-tasks)

# Fitness is a weighted blend of the judge's four 0-10 scores.
WEIGHTS = {"cynicism": 0.30, "insight": 0.35, "concision": 0.15, "format": 0.20}

OUT_PROMPT = Path(__file__).with_name("evolved_prompt.txt")
OUT_LOG = Path(__file__).with_name("evolution_log.json")
CORPUS_CACHE = Path(__file__).with_name("corpus.json")

# A tiny offline corpus so evolution runs without network access. These are
# deliberately bland wire-service style blurbs - the cynical lens has to do the
# work, which is exactly what we want to select for.
SAMPLE_ARTICLES = [
    """Title: City Council Approves Downtown Stadium Deal

The city council voted 6-3 on Tuesday to approve $400 million in public
financing for a new downtown stadium, championed by Mayor Dana Whitfield and
local development firm Crestline Partners. Whitfield called it "a generational
investment in our community's future." Crestline, whose CEO donated to the
mayor's re-election campaign, will manage construction and retain naming rights.
Opponents noted the city is cutting bus routes to close a budget gap.""",
    """Title: Tech CEO Announces Company-Wide Return to Office

Nexora's chief executive announced Monday that all employees must return to the
office five days a week starting next quarter, ending the remote policy adopted
in 2020. In a memo titled "Building Together," he cited "collaboration and
culture." The company recently signed a 15-year lease on a downtown tower and
faces pressure from investors to cut headcount. Internal surveys had shown 78%
of staff preferred hybrid work.""",
    """Title: Senator Unveils Bill to Regulate Social Media Algorithms

Senator Mara Glenn introduced legislation Wednesday that would require social
media platforms to disclose how their recommendation algorithms work. Glenn,
who is up for re-election and trailing in early polls, held the press conference
flanked by parents of teenagers. Two of the largest platforms have spent record
sums on lobbying this year. Glenn's office said the bill protects children;
critics called it unenforceable.""",
    """Title: University Renames Library After Major Donor

State University announced it will rename its main library after billionaire
alumnus Howard Pell, following his $50 million gift. The university, which has
faced criticism over rising tuition and stalled faculty contracts, said the
gift would fund "student success initiatives." Pell, who is reportedly seeking a
board seat, said he wanted to "give back to the institution that shaped me." The
naming ceremony will be held during homecoming.""",
    """Title: Airline Introduces New Boarding Fee Amid Record Profits

Continental Skies announced a new $25 "priority cabin access" fee on Thursday,
the same week it reported record quarterly profits. The airline framed the
change as "giving customers more choice." Analysts noted the fee is expected to
add $300 million in annual revenue. The CEO's compensation is tied to per-seat
revenue targets. A consumer group called it "nickel-and-diming dressed up as
flexibility.""",
]

# Each mutation call picks one of these strategies at random, to keep the
# population from collapsing toward a single style.
MUTATION_STRATEGIES = [
    "Sharpen the tone so it is even more cynical and biting.",
    "Tighten the wording for clarity and brevity without losing the edge.",
    "Swap in a different jaded persona (e.g. an ex-intelligence analyst, a "
    "burned-out investigative reporter, a disillusioned political operative).",
    "Add or refine a constraint about how each motivation must be named or "
    "justified from the article.",
    "Restructure the instructions: reorder steps or change the formatting guidance.",
    "Edit the list of human motivations: add, remove, or rename one or two of them.",
    "Inject dry, dark wit while keeping the analysis substantive.",
    "Make it demand tighter grounding in the article's actual facts.",
]


# --- Claude helpers ---------------------------------------------------------
def _text(msg) -> str:
    """Concatenate the text blocks of a Messages response."""
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def rewrite_with(prompt: str, article: str) -> str:
    """Run a candidate system prompt against one article."""
    msg = client.messages.create(
        model=GEN_MODEL,
        max_tokens=1024,
        system=prompt,
        messages=[{"role": "user", "content": article}],
    )
    return _text(msg)


JUDGE_SYSTEM = """You are a strict, consistent evaluator of "cynical news summaries".

You are given a source ARTICLE and a candidate SUMMARY that was supposed to
rewrite the article as exactly 5 bullet points, each revealing the likely real
motivation (money, power, sex, ego, fear, tribal loyalty, legacy, or similar)
of an actor in the story.

Score the SUMMARY from 0 to 10 on each axis:
- cynicism:  how sharply it exposes self-interested motives (not naive PR-speak)
- insight:   how plausible and non-obvious the inferred motivations are, given the article
- concision: how tight and punchy each bullet is (no padding)
- format:    adherence to "exactly 5 bullets, each explicitly naming a motivation"

Be discerning: reserve 9-10 for genuinely excellent work. Penalize bullets that
are vague, that invent facts not supported by the article, or that miss the
required format.

Respond with ONLY a JSON object and nothing else:
{"cynicism": <int>, "insight": <int>, "concision": <int>, "format": <int>}"""


def judge(article: str, summary: str) -> dict:
    """Score one summary. Returns the four 0-10 axes as a dict."""
    user = (
        f"ARTICLE:\n{article[:3500]}\n\n"
        f"SUMMARY TO EVALUATE:\n{summary}\n\n"
        "Score it now. JSON only."
    )
    msg = client.messages.create(
        model=META_MODEL,
        max_tokens=200,
        system=[
            # The judge rubric is identical on every call, so cache it.
            {"type": "text", "text": JUDGE_SYSTEM,
             "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": user}],
    )
    return _parse_scores(_text(msg))


def _parse_scores(text: str) -> dict:
    """Pull the JSON score object out of the judge's reply, defensively."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {k: 0 for k in WEIGHTS}
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {k: 0 for k in WEIGHTS}
    return {k: _clamp(raw.get(k, 0)) for k in WEIGHTS}


def _clamp(v) -> float:
    try:
        return max(0.0, min(10.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def mutate(prompt: str) -> str:
    """Ask Claude to mutate a prompt under a randomly chosen strategy."""
    strategy = random.choice(MUTATION_STRATEGIES)
    user = (
        "Here is a system prompt used to turn a news article into cynical, "
        "motivation-revealing bullet points:\n\n"
        f"<prompt>\n{prompt}\n</prompt>\n\n"
        f"Apply exactly this mutation: {strategy}\n\n"
        "It must remain a valid system prompt that instructs a model to rewrite "
        "a news article as bullet points that expose each actor's real "
        "motivation. Output ONLY the new system prompt - no preamble, no quotes, "
        "no commentary."
    )
    return _vary(user)


def crossover(a: str, b: str) -> str:
    """Ask Claude to recombine two parent prompts into one child."""
    user = (
        "Combine the strongest elements of these two system prompts into a "
        "single coherent new system prompt. Inherit the best instructions, tone, "
        "and structure from each parent; do not just concatenate them.\n\n"
        f"<prompt_a>\n{a}\n</prompt_a>\n\n"
        f"<prompt_b>\n{b}\n</prompt_b>\n\n"
        "It must remain a valid system prompt that instructs a model to rewrite "
        "a news article as bullet points exposing each actor's real motivation. "
        "Output ONLY the new system prompt - no preamble, no quotes, no commentary."
    )
    return _vary(user)


VARY_SYSTEM = (
    "You are a prompt engineer running a genetic algorithm over system prompts. "
    "You produce variant prompts on demand. You always reply with the prompt "
    "text only - never any explanation, quotation marks, or markdown fences."
)


def _vary(user: str) -> str:
    msg = client.messages.create(
        model=META_MODEL,
        max_tokens=1024,
        temperature=1.0,
        system=[{"type": "text", "text": VARY_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    return _clean_prompt(_text(msg))


def _clean_prompt(out: str) -> str:
    """Strip wrappers a model may add despite being told not to.

    Handles markdown fences, surrounding single/triple quotes, and a stray
    trailing close-paren from a model that echoed the prompt as a Python string
    literal (e.g. ending in a quote followed by a paren).
    """
    out = out.strip()
    out = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", out).strip()
    # A model that echoed the prompt as a Python string may close it with
    # `""")` or `')`; drop that trailing paren so the quote-peel below catches it.
    out = re.sub(r"""(['"])\)\s*$""", r"\1", out)
    # Peel matching triple-quote wrappers, then single/double-quote wrappers.
    for q in ('"""', "'''", '"', "'"):
        if len(out) >= 2 * len(q) and out.startswith(q) and out.endswith(q):
            out = out[len(q):-len(q)].strip()
            break
    return out


# --- GA core ----------------------------------------------------------------
@dataclass
class Individual:
    prompt: str
    fitness: float | None = None          # None => not yet evaluated
    scores: dict = field(default_factory=dict)  # averaged judge axes


def fitness_of(scores: dict) -> float:
    """Blend the four 0-10 axes into a single 0-1 fitness."""
    return sum(WEIGHTS[k] * scores.get(k, 0) for k in WEIGHTS) / 10.0


def evaluate(ind: Individual, articles: list[str]) -> None:
    """Run a candidate over every eval article and average the judge scores."""
    totals = {k: 0.0 for k in WEIGHTS}
    for article in articles:
        try:
            summary = rewrite_with(ind.prompt, article)
            scores = judge(article, summary)
        except Exception as e:  # one flaky call shouldn't kill the run
            print(f"    ! eval error: {e}")
            scores = {k: 0 for k in WEIGHTS}
        for k in WEIGHTS:
            totals[k] += scores[k]
    n = len(articles)
    ind.scores = {k: totals[k] / n for k in WEIGHTS}
    ind.fitness = fitness_of(ind.scores)


def tournament(pop: list[Individual], k: int) -> Individual:
    """Pick the fittest of k random contestants."""
    return max(random.sample(pop, min(k, len(pop))), key=lambda i: i.fitness)


def initial_population(size: int) -> list[Individual]:
    """Seed gen 0 with the live prompt plus mutated copies of it."""
    pop = [Individual(prompt=SYSTEM_PROMPT.strip())]
    while len(pop) < size:
        try:
            pop.append(Individual(prompt=mutate(SYSTEM_PROMPT.strip())))
        except Exception as e:
            print(f"  ! seed mutation failed: {e}")
            pop.append(Individual(prompt=SYSTEM_PROMPT.strip()))
    return pop


def breed(pop: list[Individual], size: int, elite: int,
          tourn_k: int, cx_rate: float, mut_rate: float) -> list[Individual]:
    """Produce the next generation: elites carried over, rest bred."""
    ranked = sorted(pop, key=lambda i: i.fitness, reverse=True)
    nxt = ranked[:elite]  # elitism: keep fitness so they aren't re-evaluated
    while len(nxt) < size:
        p1 = tournament(pop, tourn_k)
        try:
            if random.random() < cx_rate:
                p2 = tournament(pop, tourn_k)
                child = crossover(p1.prompt, p2.prompt)
            else:
                child = p1.prompt
            if random.random() < mut_rate or child == p1.prompt:
                child = mutate(child)
        except Exception as e:
            print(f"  ! breeding error: {e}")
            child = p1.prompt
        nxt.append(Individual(prompt=child))
    return nxt


# --- corpus -----------------------------------------------------------------
def load_corpus(urls_file: str | None) -> list[str]:
    if not urls_file:
        return SAMPLE_ARTICLES
    if CORPUS_CACHE.exists():
        cached = json.loads(CORPUS_CACHE.read_text())
        if cached:
            print(f"Loaded {len(cached)} cached articles from {CORPUS_CACHE.name}")
            return cached
    urls = [u.strip() for u in Path(urls_file).read_text().splitlines()
            if u.strip() and not u.startswith("#")]
    articles = []
    for url in urls:
        try:
            print(f"Fetching {url} ...")
            articles.append(fetch_article(url))
        except Exception as e:
            print(f"  ! skipped {url}: {e}")
    if not articles:
        print("No articles fetched; falling back to built-in samples.")
        return SAMPLE_ARTICLES
    CORPUS_CACHE.write_text(json.dumps(articles))
    return articles


# --- driver -----------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Evolve the cynical-news system prompt.")
    ap.add_argument("--pop", type=int, default=6, help="population size")
    ap.add_argument("--gens", type=int, default=3, help="number of generations")
    ap.add_argument("--eval-size", type=int, default=2,
                    help="articles sampled (once) to score every candidate")
    ap.add_argument("--elite", type=int, default=2, help="elites kept each generation")
    ap.add_argument("--tournament", type=int, default=3, help="tournament size")
    ap.add_argument("--cx-rate", type=float, default=0.6, help="crossover probability")
    ap.add_argument("--mut-rate", type=float, default=0.7, help="extra-mutation probability")
    ap.add_argument("--urls", help="file of article URLs to build/cache a corpus")
    ap.add_argument("--seed", type=int, help="RNG seed for reproducibility")
    ap.add_argument("--gen-model", default=GEN_MODEL,
                    help=f"model that runs each candidate prompt (default {GEN_MODEL})")
    ap.add_argument("--meta-model", default=META_MODEL,
                    help=f"model for mutate/crossover/judge (default {META_MODEL})")
    args = ap.parse_args()

    # Apply any model overrides to the module-level globals the helpers read.
    globals()["GEN_MODEL"] = args.gen_model
    globals()["META_MODEL"] = args.meta_model

    if args.seed is not None:
        random.seed(args.seed)
    args.elite = min(args.elite, args.pop)

    corpus = load_corpus(args.urls)
    eval_set = random.sample(corpus, min(args.eval_size, len(corpus)))
    print(f"\nEvolving: pop={args.pop} gens={args.gens} "
          f"eval_set={len(eval_set)} article(s)\n")

    pop = initial_population(args.pop)
    history = []
    best = None

    for gen in range(args.gens):
        print(f"--- Generation {gen} ---")
        for i, ind in enumerate(pop):
            if ind.fitness is None:           # elites are already scored
                evaluate(ind, eval_set)
            print(f"  [{i}] fitness={ind.fitness:.3f}  {ind.scores}")

        gen_best = max(pop, key=lambda i: i.fitness)
        if best is None or gen_best.fitness > best.fitness:
            best = gen_best
        print(f"  => best this gen: {gen_best.fitness:.3f} | "
              f"best overall: {best.fitness:.3f}\n")
        history.append({
            "generation": gen,
            "best_fitness": gen_best.fitness,
            "best_scores": gen_best.scores,
            "population": [{"fitness": i.fitness, "scores": i.scores,
                            "prompt": i.prompt} for i in pop],
        })

        if gen < args.gens - 1:
            pop = breed(pop, args.pop, args.elite, args.tournament,
                        args.cx_rate, args.mut_rate)

    OUT_PROMPT.write_text(best.prompt + "\n")
    OUT_LOG.write_text(json.dumps({
        "best_fitness": best.fitness,
        "best_scores": best.scores,
        "best_prompt": best.prompt,
        "config": vars(args),
        "history": history,
    }, indent=2))

    print("=" * 70)
    print(f"Best fitness: {best.fitness:.3f}  scores={best.scores}")
    print(f"Best prompt written to: {OUT_PROMPT.name}")
    print(f"Full log written to:    {OUT_LOG.name}")
    print("=" * 70)
    print("\n--- BEST PROMPT ---\n")
    print(best.prompt)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
