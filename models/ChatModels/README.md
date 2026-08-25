# ChatModels

One interface (`.invoke()`), four LLM providers underneath — hosted and local. This folder is where I learned that "LangChain gives you a unified interface" is true for the *call*, but not for the *setup*: every provider still has its own auth model, kwargs, and quirks.

## Concept: what is a `ChatModel`?

In LangChain, a `ChatModel` is a wrapper class that standardizes how you talk to a conversational LLM. Regardless of provider, you construct a model object and call `model.invoke(prompt)`, and get back an `AIMessage` object with a `.content` attribute holding the text. That consistency is the entire value proposition — it's what lets you swap `ChatOpenAI` for `ChatAnthropic` without rewriting your application logic.

What it does *not* standardize: how you authenticate, what parameters control randomness/length, or where the model actually runs.

## Files

### `demo.py` — OpenAI
```python
model = ChatOpenAI(model='gpt-4', temperature=1.5)
```
- Provider: OpenAI, via `langchain_openai`
- `temperature=1.5` — deliberately high, to see more creative/varied output (used here for a poem-writing prompt)
- Auth: `OPENAI_API_KEY` from `.env`

### `chatModel_anthropic.py` — Anthropic
```python
model = ChatAnthropic(model='claude-3.5-sonnet-20241012', temperature=1.0, max_completion_tokens=50)
```
- Provider: Anthropic, via `langchain_anthropic`
- `max_completion_tokens=50` — hard cap on response length, useful for keeping test calls cheap
- Auth: `ANTHROPIC_API_KEY` from `.env`

### `chatModel_google.py` — Google
```python
model = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')
```
- Provider: Google, via `langchain_google_genai`
- No explicit temperature set — uses provider default
- Auth: `GOOGLE_API_KEY` from `.env`

### `chatModel_huggingFace.py` — HuggingFace, hosted
```python
LLM = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", task="conversational", provider="auto", huggingfacehub_api_token=token)
model = ChatHuggingFace(llm=LLM)
```
- Provider: HuggingFace Inference Endpoint (hosted, remote inference)
- Two-step wrapping: `HuggingFaceEndpoint` handles the actual API call, `ChatHuggingFace` wraps it to expose the standard chat interface
- `provider="auto"` lets HuggingFace route to whichever backend (e.g., an inference provider partner) is available for that model
- Auth: `HUGGINGFACEHUB_API_TOKEN` from `.env`

### `chatModel_HF_Local.py` — HuggingFace, local
```python
LLM = HuggingFacePipeline(model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0', task='text-generation', pipeline_kwargs=dict(temperature=0.5, max_new_tokens=100))
model = ChatHuggingFace(llm=LLM)
```
- Provider: none — model weights download and run **on your own machine** via `transformers`
- `max_new_tokens` (not `max_completion_tokens` — different provider, different kwarg name, same underlying idea) caps generation length
- No API key, no network call at inference time (aside from the initial model download)
- Tradeoff: no cost, no rate limits, but you're bound by local compute — a 1.1B model is small enough to run on modest hardware, unlike the 7B+ hosted models above

## Key takeaway

Every script above ends in the same call shape:
```python
result = model.invoke("...")
print(result.content)
```
But getting to that point requires knowing each provider's auth mechanism, its specific rate/length-limiting kwarg names (`max_completion_tokens` vs `max_new_tokens`), and whether inference happens on someone else's server or your own. The abstraction saves you from relearning the *call*, not from understanding the *provider*.

## Run

```bash
python demo.py                    # OpenAI — needs OPENAI_API_KEY
python chatModel_anthropic.py     # Anthropic — needs ANTHROPIC_API_KEY
python chatModel_google.py        # Google Gemini — needs GOOGLE_API_KEY
python chatModel_huggingFace.py   # HuggingFace hosted — needs HUGGINGFACEHUB_API_TOKEN
python chatModel_HF_Local.py      # HuggingFace local — no key needed, downloads model on first run
```
