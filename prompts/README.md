# prompts

Part of the [`langchain`](../) repo — a hands-on exploration of how prompts work in [LangChain](https://python.langchain.com/), starting from a raw, unstructured LLM call and progressively building up to reusable, serialized, chainable prompt templates. This folder is essentially a learning path: each script isolates one new concept so the difference between approaches is easy to see side by side.

## Why this folder exists

When you start with LangChain, it's tempting to just call `model.invoke(some_string)` everywhere. That works, but it doesn't scale — prompts become hard to reuse, hard to version, and hard to keep consistent across an app. This folder walks through the fix for that, one step at a time:

1. Call the model directly with no structure.
2. Introduce variables into the prompt.
3. Save that prompt to disk so it isn't duplicated in code.
4. Load the saved prompt and compose it into a proper chain.
5. Handle multi-turn state (a chatbot), which prompt templates alone don't solve.

## Files, in learning order

### 1. `static_prompts.py` — the baseline
A Streamlit app with a single text box. Whatever the user types goes straight into `model.invoke()`.

```python
user_query = st.text_input("Enter your query")
if st.button("Summarize"):
    result = model.invoke(user_query)
    st.write(result.content)
```

**Concept:** no templating at all — this is the "before" picture. There's no consistent instruction set behind the scenes, so output quality and format depend entirely on what the user happens to type.

**Limitation:** every user has to know how to write a good prompt themselves. There's no way to enforce structure (e.g. "always include math intuition," "always use Markdown").

---

### 2. `dynamic_prompts.py` — introducing `PromptTemplate`
Same Streamlit UI pattern, but now the user picks from dropdowns (paper, tone, length) instead of typing free text, and those choices get substituted into a large, fixed instruction template via LangChain's `PromptTemplate`:

```python
template1 = PromptTemplate(
    template="""Please summarize the research paper titled "{paper_input}" ...""",
    input_variables=["paper_input", "explain_style", "length"],
)

prompt = template1.invoke({
    'paper_input': input_paper,
    'explain_style': style_input,
    'length': output_length
})

result = model.invoke(prompt)
```

**Concept:** separating the *fixed instructions* (math details, analogies, formatting rules, fidelity-to-source constraints) from the *variable inputs* (paper, tone, length). The instructions live once in the template; only the three variables change per run.

**Limitation:** the template is still hardcoded as a Python string inside this file. If another script wants the same prompt, it has to copy-paste it — which is exactly how prompts drift out of sync.

---

### 3. `prompt_template_generator.py` → `template.json` — persisting the template
This script defines the *same* `PromptTemplate` as above, but instead of using it immediately, it serializes it to disk:

```python
template = PromptTemplate(template="...", input_variables=[...])
template.save("template.json")
```

This produces `template.json`, a plain JSON file capturing the template string, its input variables, the template format (`f-string`), and metadata LangChain needs to reconstruct it later (`_type: "prompt"`).

**Concept:** treating a prompt as a **versionable artifact**, not a code fragment. `template.json` can be reviewed in a diff, shared across scripts, checked into version control independently of application logic, or even swapped out at runtime without touching Python code.

**Run this once** whenever you change the prompt text — it regenerates `template.json`, which the next script depends on.

---

### 4. `chain_prompts.py` — loading the template and chaining it
Rather than redefining the prompt, this script loads it straight from disk with `load_prompt()`, then composes it with the model using **LangChain Expression Language (LCEL)**:

```python
template = load_prompt('template.json')
model = ChatOpenAI(model='gpt-4')

chain = template | model
result = chain.invoke({
    'paper_input': input_paper,
    'explain_style': style_input,
    'length': output_length
})
```

**Concept:** the `|` pipe operator builds a `Runnable` chain — the dict of inputs flows into the template, the formatted prompt flows into the model, and `chain.invoke(...)` runs the whole pipeline in one call. This is the idiomatic LangChain pattern for anything with more than one step (template → model → parser → ...), and it's what makes it trivial to later insert extra steps (an output parser, a retriever, a second model call) without restructuring the calling code.

**This is the most "production-shaped" script in the folder** — it's the pattern you'd actually build on.

---

### 5. `chatbot.py` — multi-turn state
A plain command-line loop, deliberately separate from the templating story above, because it demonstrates a different problem: **conversational memory**.

```python
chat_history = {}
while True:
    user_input = input("Prompt: ")
    if user_input == 'Exit':
        break
    result = model.invoke(user_input)
    chat_history[user_input] = result.content
    print(f'AI:{result.content}')
```

**Concept:** each call to `model.invoke()` is stateless — the model has no memory of prior turns unless you explicitly resend the conversation. This script stores history locally in a dict, but note that it **doesn't yet feed that history back into the model** on each turn, so the model can't actually "remember" earlier exchanges — it can only be inspected afterward via `print(chat_history)`. That's a natural next thing to fix (e.g. by passing a running list of messages back into `invoke()` each turn, or reaching for LangChain's message history / memory utilities).

## How the pieces relate

```
static_prompts.py          dynamic_prompts.py
   (no template)      →     (inline PromptTemplate)
                                    │
                                    ▼
                       prompt_template_generator.py
                                    │
                                    ▼
                             template.json
                                    │
                                    ▼
                            chain_prompts.py
                         (load_prompt + LCEL chain)

chatbot.py  — standalone, illustrates multi-turn state separately
```

## Setup

```bash
pip install langchain langchain-openai streamlit python-dotenv
```

Create a `.env` file in this folder with your OpenAI API key (loaded via `python-dotenv`'s `load_dotenv()` in every script):

```
OPENAI_API_KEY=your-key-here
```

## Running things

Regenerate the template file first if you don't already have it, or if you've edited the prompt text:

```bash
python prompt_template_generator.py
```

Then run whichever Streamlit app you want to try:

```bash
streamlit run static_prompts.py
streamlit run dynamic_prompts.py
streamlit run chain_prompts.py
```

Or run the chatbot directly in the terminal:

```bash
python chatbot.py
```

Type prompts at the `Prompt:` input; type `Exit` to quit — it will print the full `chat_history` dict on exit.

## Notes & gotchas

- All scripts use `gpt-4` via `ChatOpenAI`. Change the `model=` argument if you want a different OpenAI model (e.g. `gpt-4o`, `gpt-4o-mini`).
- `chain_prompts.py` **requires `template.json` to exist** in the same folder — it will error if you haven't run `prompt_template_generator.py` yet.
- If you edit the prompt text in `dynamic_prompts.py`, remember it's a separate copy from the one in `prompt_template_generator.py` — they aren't linked. Keeping them in sync (or removing the duplication) is a good next refactor.
- `chatbot.py`'s `chat_history` dict is keyed by user input, so if you send the exact same prompt twice, the second response overwrites the first in that dict (though both still print to the console as they happen).
- None of the Streamlit scripts guard against empty input (e.g. clicking "Summarize" with no query in `static_prompts.py`) — worth adding validation if you extend these.
