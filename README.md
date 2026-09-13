# LangChain Course

A hands-on project for learning [LangChain](https://python.langchain.com/) — building prompt chains, switching between LLM providers, and working with both hosted APIs and local models.

## What's here

- **`main.py`** — A working example that builds a `PromptTemplate`, pipes it into an LLM using LangChain Expression Language (LCEL), and asks the model to summarize a block of text and pull out interesting facts. Currently configured to run against a local [Ollama](https://ollama.com/) model (`gemma3:270m`), with a commented-out `ChatOpenAI` example showing how to swap in OpenAI instead.
- **`excerise/1. ModelSwitching.py`** — A guided exercise (using mock objects) for practicing how to switch between different Groq models with LangChain, load API keys from environment variables, and query/compare model responses.

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.com/) installed and running locally, with the model pulled:
  ```bash
  ollama pull gemma3:270m
  ```
- (Optional) API keys for hosted providers you want to use instead of/alongside Ollama — e.g. `OPENAI_API_KEY`, `GROQ_API_KEY`

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sureshmanem/langchain-course.git
   cd langchain-course
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Create a `.env` file in the project root with any API keys you need:
   ```
   OPENAI_API_KEY=your-key-here
   GROQ_API_KEY=your-key-here
   ```

## Running

Run the main example:
```bash
uv run main.py
```

Run the model-switching exercise:
```bash
uv run "excerise/1. ModelSwitching.py"
```

## Dependencies

Managed via `pyproject.toml` / `uv.lock`, including:
- `langchain` — core framework
- `langchain-openai` — OpenAI integration
- `langchain-ollama` — local model integration via Ollama
- `python-dotenv` — loads `.env` files
- `deepagents` — agent-building utilities
- `black`, `isort` — formatting/linting
