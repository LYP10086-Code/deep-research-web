# Deep Research From Scratch - Repository Guide

## Repository Structure

This repository builds a comprehensive deep research system from scratch using LangGraph, progressing through 5 tutorial notebooks that demonstrate different components and patterns.

```
deep_research_from_scratch/
├── frontend/               # Web frontend (Vite + React + TS + Tailwind)
│   └── src/
│       ├── pages/          # ChatPage / ReportsPage / ReportDetailPage
│       ├── components/     # Sidebar
│       ├── api.ts          # REST + SSE client
│       └── store.ts        # Zustand store
├── backend/                # FastAPI server wrapping the research graph
│   ├── graph_def.py        # Graph construction (extracted from notebooks/5_full_agent copy.py)
│   ├── runner.py           # Per-session asyncio task: astream -> interrupt -> resume
│   ├── db.py               # SQLite business DB (sessions / messages / reports)
│   ├── main.py             # FastAPI app: REST + SSE (uvicorn backend.main:app)
│   └── data/               # Runtime data (gitignored)
├── research/               # Original tutorial project (source of truth)
│   ├── notebooks/          # Interactive tutorial notebooks (MODIFY THESE)
│   │   ├── 1_scoping.ipynb          # User clarification and brief generation
│   │   ├── 2_research_agent.ipynb   # Research agent with custom tools
│   │   ├── 3_research_agent_mcp.ipynb  # Research agent with MCP servers
│   │   ├── 4_research_supervisor.ipynb # Multi-agent supervisor coordination
│   │   ├── 5_full_agent.ipynb       # Complete end-to-end system
│   │   └── utils.py                 # Shared utilities for notebooks
│   ├── src/deep_research_from_scratch/  # Generated source code (DO NOT MODIFY)
│   │   ├── multi_agent_supervisor.py
│   │   ├── prompts.py
│   │   ├── research_agent.py
│   │   ├── research_agent_mcp.py
│   │   ├── state_*.py
│   │   └── utils.py
│   ├── pyproject.toml      # Package definition (pip install -e ./research)
│   └── langgraph.json
├── .env                    # API keys (loaded by backend and scripts)
├── AGENTS.md               # This guide
└── README.md               # Project overview: web system usage + link to research/ tutorial docs
```

## 🚨 Important Development Workflow

**The notebooks in `research/notebooks/` are the source of truth and should be the ONLY files modified.**

The source code in `research/src/deep_research_from_scratch/` is automatically generated from the notebooks using `%%writefile` magic commands. Here's how it works:

### How Code Generation Works

1. **Notebooks contain `%%writefile` cells**: Each notebook uses Jupyter's `%%writefile` magic to write code directly to files in `src/`
2. **Notebooks are executable tutorials**: They demonstrate concepts interactively while generating the production code
3. **Source files are generated artifacts**: The `.py` files in `src/` are outputs, not inputs

### Example from notebooks:
```python
%%writefile ../src/deep_research_from_scratch/research_agent.py

"""
Research Agent Implementation
"""
# ... actual implementation code follows
```

### Development Guidelines

- ✅ **DO**: Edit notebooks in `research/notebooks/` directory
- ✅ **DO**: Run notebook cells to regenerate source code
- ✅ **DO**: Test changes by running the notebooks
- ❌ **DON'T**: Directly edit files in `research/src/deep_research_from_scratch/`
- ❌ **DON'T**: Expect manual changes to `research/src/` files to persist

## System Architecture

The system implements a three-phase deep research workflow:

1. **Scope** (Notebook 1): Clarify research scope and generate structured briefs
2. **Research** (Notebooks 2-4): Perform research using various agent patterns
3. **Write** (Notebook 5): Synthesize findings into comprehensive reports

### Key Components

- **Scoping Agent**: Clarifies user intent and generates research briefs
- **Research Agent**: Iterative research with custom tools or MCP servers
- **Supervisor Agent**: Coordinates multiple research agents for complex topics
- **Full System**: Integrates all components into end-to-end workflow

## Quick Start for Development

1. Make changes to the appropriate notebook in `research/notebooks/`
2. Run the modified cells to regenerate source code
3. Test the changes by running subsequent notebook cells
4. The generated code in `research/src/` will automatically reflect your changes

This approach ensures that the interactive tutorials remain the authoritative source while automatically maintaining the corresponding Python package structure.

## Code Quality and Formatting

### Ruff Formatting Checks

To maintain consistent code formatting across the generated source files, run ruff periodically:

```bash
# Check for formatting issues
ruff check research/src/

# Auto-fix formatting issues where possible
ruff check research/src/ --fix

# Check specific file
ruff check research/src/deep_research_from_scratch/research_agent.py
```

**Important**: Since source files in `research/src/` are generated from notebooks, any formatting issues should be fixed in the notebook `%%writefile` cells, not directly in the source files. After fixing formatting in notebooks, regenerate the source files by running the notebook cells.

**Common formatting fixes needed:**
- **D212**: Ensure docstring summaries start on the same line as triple quotes
- **I001**: Organize imports properly (standard library → third party → local imports)
- **F401**: Remove unused imports
- **D415**: Add periods to docstring summaries