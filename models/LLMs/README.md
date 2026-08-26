# LLMs

The base, non-chat completion interface — kept in its own folder to keep the `LLM` vs `ChatModel` distinction explicit rather than something that quietly blurs together.

## Concept: `LLM` vs `ChatModel`

LangChain has two parallel abstractions for talking to language models:

| | `LLM` class | `ChatModel` class |
|---|---|---|
| Input | plain string | list of messages (or a string, wrapped) |
| Output | plain string | `AIMessage` object (`.content` holds the text) |
| Mental model | "complete this text" | "respond in a conversation" |
| Example here | `demo.py` (`OpenAI`, `gpt-3.5-turbo-instruct`) | everything in `ChatModels/` |

Older/base models (like instruct-tuned completion models) are typically wrapped with the `LLM` class. Modern conversational models (GPT-4, Claude, Gemini) are wrapped with `ChatModel` classes, since they're trained specifically to handle multi-turn, role-based conversation (system/user/assistant), not raw text continuation.

## Files

### `demo.py`
```python
llm = OpenAI(model='gpt-3.5-turbo-instruct')
result = llm.invoke("What is the capital of Pakistan?")
print(f'{result}')
```
- `gpt-3.5-turbo-instruct` is a completion-style model — no chat formatting, no message roles, just "continue this text"
- `llm.invoke(str)` returns a plain string directly (no `.content` unwrapping needed, unlike `ChatModel.invoke()`)

## Why this distinction matters

Mixing up `LLM` and `ChatModel` interfaces is a common source of bugs when switching models — code written for one won't work unmodified against the other, since the input/output shapes differ. Isolating this here made the difference concrete instead of something I'd just read about.

## Run

```bash
python demo.py
```
Requires `.env` with `OPENAI_API_KEY`.
