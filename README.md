# Docs Q&A (LangChain)

A small **retrieval-augmented generation (RAG)** app. Point it at a folder of Markdown or text notes, ask a question, and Claude answers using only those notes, with the source files cited.

Built with **LangChain Expression Language (LCEL)**, **Claude** (via `langchain-anthropic`) and **BM25** keyword retrieval. No vector database or embeddings API required.

---

## What problem does it solve?

Language models don't know what's in your private notes, and when they don't know, they tend to guess. This app grounds the model in your own documents:

- It retrieves only the passages relevant to your question.
- It instructs the model to answer **only** from those passages, and to say "I don't know" otherwise.
- It tells you which files the answer came from, so you can verify it.

---

## How it works

```
question ──▶ BM25 retriever ──▶ top-k chunks ──▶ prompt ──▶ Claude ──▶ answer + sources
              (custom LangChain                  (answer only
               BaseRetriever)                     from context)
```

1. **Load.** `load_docs` reads every `.md` and `.txt` file in a folder and records each file name as metadata.
2. **Split.** `RecursiveCharacterTextSplitter` cuts documents into overlapping 500-character chunks so retrieval is precise.
3. **Retrieve.** A custom `BM25Retriever` (a subclass of LangChain's `BaseRetriever`) scores each chunk against the question using the BM25 ranking algorithm and returns the top *k*.
4. **Generate.** An LCEL chain formats the chunks into a prompt and sends it to Claude with temperature 0.
5. **Return.** The output is a dictionary: `{"answer": "...", "sources": ["postgres.md"]}`.

### The chain in code

```python
RunnableParallel(docs=itemgetter("question") | retriever, question=itemgetter("question"))
    | RunnablePassthrough.assign(answer=generate)   # prompt | llm | StrOutputParser
    | RunnableLambda(to_answer_and_sources)
```

---

## Design decisions

- **BM25 instead of embeddings.** Keyword ranking works well for notes and documentation, needs no extra API key, and keeps the project easy to run. The retriever is a drop-in component, so it can be swapped for a vector store later.
- **Custom retriever.** `langchain-community` is being sunset, so the retriever is implemented directly on `rank_bm25` as a `BaseRetriever` subclass. This removes a dependency and shows how LangChain's retriever interface works.
- **Grounded prompt.** The system prompt restricts answers to the supplied context and gives the model an explicit "I don't know" fallback to reduce hallucination.
- **Source citation.** Sources are de-duplicated and sorted from retrieved chunk metadata, so every answer is traceable.
- **Zero-score filtering.** Chunks with no keyword overlap are dropped rather than padded into the context.
- **Offline-testable.** The LLM is injected into `build_chain`, so tests can use a fake model.

---

## Project structure

```
docs-qa/
├── pyproject.toml
├── requirements.txt
├── README.md
├── docs/                    # sample notes (docker, postgres, git)
├── src/docs_qa/
│   ├── rag.py               # loader, retriever, prompt, LCEL chain
│   ├── cli.py               # command-line interface
│   └── __main__.py          # enables `python -m docs_qa`
└── tests/
    └── test_rag.py          # offline tests
```

---

## Setup

Requires **Python 3.10+** and an [Anthropic API key](https://console.anthropic.com).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...   # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Usage

```bash
# One-shot question against the sample notes
python -m docs_qa "How do I index soft-deleted rows?"

# Interactive mode
python -m docs_qa

# Your own notes folder, retrieving 5 chunks
python -m docs_qa --docs ~/notes -k 5 "What's our git workflow?"
```

Example output:

```
$ python -m docs_qa "How do I index soft-deleted rows?"

Use a partial index on rows where deleted_at IS NULL, and confirm it is used with EXPLAIN ANALYZE.

Sources: postgres.md
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `question` | *(interactive)* | Question to ask; omit to enter interactive mode |
| `--docs` | `docs` | Folder of `.md` / `.txt` files |
| `-k` | `3` | Number of chunks to retrieve |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `DOCS_QA_MODEL` | `claude-sonnet-5-5` | Claude model to use |

---

## Testing

```bash
pytest
```

The tests run offline with LangChain's `FakeListChatModel`, so no API key is needed. They verify that:

- A missing folder raises a clear error
- Retrieval ranks the correct file first for a given query
- The chain returns the expected `answer` and `sources` shape
- Retrieved context actually reaches the prompt sent to the model (using a spy model)

---

## Tech stack

- Python 3.10+
- [LangChain](https://python.langchain.com) (`langchain-core`, `langchain-text-splitters`)
- `langchain-anthropic` for Claude
- `rank_bm25` for retrieval
- pytest

---

## Ideas to extend

- Semantic search with embeddings and a vector store (Chroma or FAISS)
- Conversation memory for follow-up questions
- Streaming tokens to the terminal
- A Streamlit or FastAPI front end
- PDF and DOCX loaders

---

## License

MIT
