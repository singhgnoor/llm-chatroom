# llm-chatroom

Put two LLMs in a chat room, then watch them talk to each other live in your terminal.

Built for curiosity: what do models say when no human is steering? Every run is saved, so you can analyse the conversations later.

## Quick start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python chatroom.py
```

Pick a topic, models and length:

```bash
python chatroom.py --topic "Do you two experience anything?" --turns 16
python chatroom.py --model-a gpt-6-luna --model-b gpt-5.6-luna
python chatroom.py --persona-b "You are {name}, a stubborn skeptic arguing with {other}. Keep it to 2-3 sentences."
```

Press `Ctrl+C` to stop at any time. The transcript is still saved.

## Spending limits

The script is built so it can't run up your bill:

| Limit | Default | Hard ceiling (in code) |
|---|---|---|
| Budget per run | $0.08 | $0.10 |
| Turns | 20 | 50 |
| Tokens per reply | 250 | 600 |
| History each model sees | last 12 messages | none |

- **Before every call**, the script works out the *worst-case* cost of that call (an overestimate of the input plus the full reply limit). If that could take the total over budget, it stops. This makes the budget a hard ceiling, not a target.
- Cost is tracked from the real token counts that OpenAI returns.
- Command-line flags can only *lower* the ceilings. To raise one, edit the constants at the top of `chatroom.py`.
- Models without a listed price are refused, so the budget maths never guesses.
- API errors end the run. There is at most one automatic retry.

For scale: a 20-turn chat on `gpt-6-luna` costs roughly **a tenth of a cent**.

Prices live in the `PRICES` dict and were checked in Nov 2026. If OpenAI changes them, update the dict.

## Output

Each run writes two files to `transcripts/` (git-ignored):

- `chat-YYYYMMDD-HHMMSS.txt` to read
- `chat-YYYYMMDD-HHMMSS.jsonl` to analyse: one config line, one line per turn (speaker, model, text, tokens, cost, latency) and an end line

Quick look with pandas:

```python
import pandas as pd
df = pd.read_json("transcripts/chat-....jsonl", lines=True)
turns = df[df.type == "turn"]
print(turns[["speaker", "tokens_out", "cost_usd"]])
```

## Models

USD per 1M tokens (Nov 2026). Stick to the cheap rows for long chats; the big models burn through the budget in a few turns.

| Model | Input | Output  |
|---|---|---|
| `gpt-6-luna` (default) | 0.10 | 0.50 |
| `gpt-5.6-luna` | 0.20 | 1.20 |
| `gpt-6.1-terra` | 1.00 | 5.00 |
| `gpt-6-sol` | 2.00 | 10.00 |
| `gpt-6.1-sol` | 2.00 | 10.00 |
| `gpt-6.2-sol` | 2.00 | 10.00 |
| `gpt-5.6-terra` | 2.00 | 12.00 |
| `gpt-5.6-sol` | 4.00 | 20.00 |
| `gpt-6.2-astra` | 8.00 | 40.00 |
| `gpt-6-astra` | 10.00 | 50.00 |

To be safe, the budget maths charges every input token at the 30-minute cache-write rate, which is higher than the normal input rate, and ignores cache-read discounts. Your real bill should come in a little *under* what the script shows.

If a model thinks before answering and you get empty replies, its thinking used up the reply limit. Raise it with `--max-tokens 500`.