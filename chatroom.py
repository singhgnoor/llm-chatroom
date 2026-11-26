#!/usr/bin/env python3
"""
llm-chatroom: put two LLMs in a room and watch them talk.

- Streams the conversation live in your terminal.
- Saves every run to transcripts/ (.txt to read, .jsonl to analyse).
- Hard spending cap: before each call it works out the WORST-CASE cost of
  that call, and stops if that could push the total over the budget.
  So the run can never go over budget, as long as the prices below are right.

Usage:
    export OPENAI_API_KEY=sk-...
    python chatroom.py
    python chatroom.py --topic "Is math invented or discovered?" --turns 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Safety limits. The command line can lower these but never raise them.
# To raise one, edit it here on purpose.

HARD_BUDGET_USD = 0.10      # absolute ceiling for one run
MAX_TURNS = 20              # absolute ceiling on messages in one run
MAX_REPLY_TOKENS = 600      # absolute ceiling on one reply's length
HISTORY_WINDOW = 12         # each model sees only the last N messages

# USD per 1M tokens, from https://openai.com/api/pricing (Nov 2026):
#   model: (input, output, cached_input, cache_write_30m, cache_write_12h)
# Models not listed here are refused, so the budget maths never guesses.
PRICES = {
    # "gpt-6.2-astra": (8.00, 40.00, 0.40, 10.00, 20.00),
    # "gpt-6.2-sol":   (2.00, 10.00, 0.10,  2.50,  5.00),
    # "gpt-6-astra":   (10.00, 50.00, 1.00, 12.50, 25.00),
    # "gpt-6.1-sol":   (2.00, 10.00, 0.10,  2.50,  5.00),
    # "gpt-6.1-terra": (1.00,  5.00, 0.05,  1.25,  2.50),
    "gpt-6-luna":    (0.10,  0.50, 0.01, 0.125,  0.25),
    # "gpt-6-sol":     (2.00, 10.00, 0.20,  2.50,  5.00),
    # "gpt-5.6-sol":   (4.00, 20.00, 0.40,  5.00, 10.00),
    # "gpt-5.6-terra": (2.00, 12.00, 0.20,  2.50,  5.00),
    "gpt-5.6-luna":  (0.20,  1.20, 0.02,  0.25,  0.50),
}
# Hidden "thinking" tokens are billed as output and count toward
# max_completion_tokens, so the per-reply cap still holds for any model.

DEFAULT_PERSONA = (
    "You are {name}, an AI talking with another AI called {other} in a chat "
    "room. No human is taking part. Be yourself, be curious, and keep each "
    "reply short: 2-4 sentences. Don't wrap up or say goodbye unless you "
    "really want to end the conversation."
)

# ANSI colours (switched off when output isn't a terminal)
_TTY = sys.stdout.isatty()
def _c(code: str) -> str:
    return code if _TTY else ""
RESET, DIM, BOLD = _c("\033[0m"), _c("\033[2m"), _c("\033[1m")
COLORS = [_c("\033[96m"), _c("\033[95m")]  # cyan, magenta


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Deliberately pessimistic: every input token is charged at the
    30-minute cache-write rate (the highest rate that can apply without
    opting in to 12h caching), and cache-read discounts are ignored.
    So the real bill is at most this, usually a bit less."""
    p_in, p_out, _cached, write_30m, _write_12h = PRICES[model]
    return (tokens_in * max(p_in, write_30m) + tokens_out * p_out) / 1_000_000


def estimate_tokens(messages: list[dict]) -> int:
    """Rough and deliberately high: ~3 chars/token plus per-message overhead."""
    return sum(len(m["content"]) // 3 + 8 for m in messages) + 16


class Agent:
    def __init__(self, name: str, model: str, persona: str, other: str):
        self.name, self.model = name, model
        self.system = persona.format(name=name, other=other)

    def view(self, transcript: list[dict]) -> list[dict]:
        """The chat from this agent's point of view: its lines are
        'assistant', the other agent's lines are 'user'."""
        msgs = [{"role": "system", "content": self.system}]
        for turn in transcript[-HISTORY_WINDOW:]:
            role = "assistant" if turn["speaker"] == self.name else "user"
            msgs.append({"role": role, "content": turn["text"]})
        return msgs


class Logger:
    """Writes the run to disk as it happens, so a crash or Ctrl+C keeps it."""

    def __init__(self, out_dir: Path, config: dict):
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base, i = stamp, 1
        while (out_dir / f"chat-{stamp}.txt").exists():   # never overwrite
            i += 1
            stamp = f"{base}-{i}"
        self.txt_path = out_dir / f"chat-{stamp}.txt"
        self.jsonl_path = out_dir / f"chat-{stamp}.jsonl"
        self.txt = self.txt_path.open("w", encoding="utf-8")
        self.jsonl = self.jsonl_path.open("w", encoding="utf-8")

        self.txt.write("# llm-chatroom transcript\n")
        for k, v in config.items():
            self.txt.write(f"# {k}: {v}\n")
        self.txt.write("\n")
        self._json({"type": "config", **config})

    def _json(self, obj: dict) -> None:
        self.jsonl.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.jsonl.flush()

    def turn(self, rec: dict) -> None:
        self.txt.write(f"[{rec['turn']}] {rec['speaker']} ({rec['model']}):\n")
        self.txt.write(rec["text"].strip() + "\n\n")
        self.txt.flush()
        self._json({"type": "turn", **rec})

    def close(self, reason: str, spent: float, turns: int) -> None:
        self.txt.write(f"# ended: {reason}\n# turns: {turns}\n"
                       f"# total cost: ${spent:.5f}\n")
        self._json({"type": "end", "reason": reason,
                    "turns": turns, "cost_usd": round(spent, 6)})
        self.txt.close()
        self.jsonl.close()


def stream_reply(client, agent: Agent, messages: list[dict], max_tokens: int):
    """Streams one reply to the terminal. Returns (text, tokens_in, tokens_out)."""
    kwargs = dict(
        model=agent.model,
        messages=messages,
        max_completion_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    parts, usage = [], None
    for chunk in client.chat.completions.create(**kwargs):
        if chunk.usage:
            usage = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            piece = chunk.choices[0].delta.content
            parts.append(piece)
            sys.stdout.write(piece)
            sys.stdout.flush()

    text = "".join(parts)
    if usage:
        return text, usage.prompt_tokens, usage.completion_tokens
    # No usage reported: assume the worst so the budget stays safe.
    return text, estimate_tokens(messages), max_tokens


def run(args, client) -> None:
    for m in (args.model_a, args.model_b):
        if m not in PRICES:
            sys.exit(f"Unknown model '{m}'. Add its price to PRICES in "
                     f"chatroom.py first (known: {', '.join(PRICES)}).")

    budget = min(args.budget, HARD_BUDGET_USD)
    turns_max = min(args.turns, MAX_TURNS)
    max_tokens = min(args.max_tokens, MAX_REPLY_TOKENS)

    a = Agent(args.name_a, args.model_a, args.persona_a, args.name_b)
    b = Agent(args.name_b, args.model_b, args.persona_b, args.name_a)
    agents = [a, b]

    config = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "topic": args.topic, "budget_usd": budget, "max_turns": turns_max,
        "max_reply_tokens": max_tokens,
        "agent_a": f"{a.name} ({a.model})", "agent_b": f"{b.name} ({b.model})",
        "persona_a": a.system, "persona_b": b.system,
    }
    log = Logger(Path(args.out), config)

    print(f"{BOLD}llm-chatroom{RESET}  {COLORS[0]}{a.name}{RESET} ({a.model})"
          f"  ×  {COLORS[1]}{b.name}{RESET} ({b.model})")
    print(f"{DIM}budget ${budget:.4g} · up to {turns_max} turns · "
          f"Ctrl+C to stop{RESET}")
    print(f"{DIM}topic: {args.topic}{RESET}\n")

    # The opening line is from a moderator, so B's first reply has something
    # to react to and A sees it as the start of the chat.
    transcript = [{"speaker": "Moderator", "text": args.topic}]
    print(f"{BOLD}Moderator{RESET}\n{args.topic}\n")

    spent, n, reason = 0.0, 0, "max turns reached"
    try:
        while n < turns_max:
            agent = agents[n % 2]
            msgs = agent.view(transcript)

            # ---- hard budget gate: worst case of the NEXT call ----
            worst = cost_usd(agent.model, estimate_tokens(msgs), max_tokens)
            if spent + worst > budget:
                reason = (f"budget guard: next call could cost up to "
                          f"${worst:.5f}, only ${budget - spent:.5f} left")
                break

            color = COLORS[n % 2]
            print(f"{BOLD}{color}{agent.name}{RESET} {DIM}{agent.model}{RESET}")
            sys.stdout.write(color)
            t0 = time.time()
            text, t_in, t_out = stream_reply(client, agent, msgs, max_tokens)
            sys.stdout.write(RESET + "\n")

            c = cost_usd(agent.model, t_in, t_out)
            spent += c
            n += 1
            if not text.strip():
                text = "(no reply - likely used its whole token limit thinking)"
                print(f"{DIM}{text}{RESET}")

            rec = {"turn": n, "speaker": agent.name, "model": agent.model,
                   "text": text, "tokens_in": t_in, "tokens_out": t_out,
                   "cost_usd": round(c, 6), "latency_s": round(time.time() - t0, 2)}
            transcript.append({"speaker": agent.name, "text": text})
            log.turn(rec)

            print(f"{DIM}  turn {n}/{turns_max} · {t_in}+{t_out} tok · "
                  f"${spent:.5f} of ${budget:.4g}{RESET}\n")
            time.sleep(args.pause)
    except KeyboardInterrupt:
        sys.stdout.write(RESET)
        reason = "stopped by you (Ctrl+C)"
    except Exception as e:  # API errors etc. Stop; never retry in a loop.
        sys.stdout.write(RESET)
        reason = f"error: {type(e).__name__}: {e}"
    finally:
        log.close(reason, spent, n)

    print(f"\n{BOLD}Ended:{RESET} {reason}")
    print(f"{BOLD}Spent:{RESET} ${spent:.5f} over {n} turns")
    print(f"{BOLD}Saved:{RESET} {log.txt_path}\n       {log.jsonl_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Two LLMs, one chat room.")
    p.add_argument("--topic", default="You two have just met. Say hi and "
                   "talk about whatever you find interesting.")
    p.add_argument("--model-a", default="gpt-6-luna")
    p.add_argument("--model-b", default="gpt-6-luna")
    p.add_argument("--name-a", default="Ada")
    p.add_argument("--name-b", default="Turing")
    p.add_argument("--persona-a", default=DEFAULT_PERSONA,
                   help="System prompt for A. {name} and {other} get filled in.")
    p.add_argument("--persona-b", default=DEFAULT_PERSONA)
    p.add_argument("--turns", type=int, default=20)
    p.add_argument("--budget", type=float, default=0.08,
                   help=f"USD for this run (capped at {HARD_BUDGET_USD}).")
    p.add_argument("--max-tokens", type=int, default=250,
                   help="Longest single reply, in tokens.")
    p.add_argument("--pause", type=float, default=0.6,
                   help="Seconds between turns, for readability.")
    p.add_argument("--out", default="transcripts")
    return p.parse_args(argv)


def main():
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Missing dependency: pip install -r requirements.txt")
    # max_retries=1: one quick retry on a network blip, never a retry storm.
    run(args, OpenAI(max_retries=1, timeout=60))


if __name__ == "__main__":
    main()