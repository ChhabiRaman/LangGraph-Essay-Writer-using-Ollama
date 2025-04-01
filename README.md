# 📝 Essay Writer

A sophisticated AI-powered essay generation system built with LangGraph that can plan, research, write, and iteratively improve essays.

## ✨ Features

- 🧠 **AI-Powered Planning**: Automatically generates structured essay outlines
- 🔍 **Integrated Research**: Conducts web searches for relevant information
- ✍️ **Iterative Drafting**: Creates polished essay drafts
- 🔄 **Self-Improvement**: Critiques and refines essays through multiple revisions
- 📊 **Workflow Visualization**: Implemented as a LangGraph state machine

## 🛠️ Architecture

The workflow follows these steps:
1. **Planner**: Creates an essay outline based on the provided task
2. **Research Plan**: Conducts initial research to gather information
3. **Generate**: Writes an essay draft using the outline and research
4. **Condition**: Checks if we've reached maximum revisions
   - If max revisions reached: END
   - If revisions remain: Continue
5. **Critique**: Evaluates the essay and provides feedback
6. **Research Critique**: Gathers additional information based on critique
7. The system then loops back to **Generate** for revision and improvement

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) with access to llama3.2 model
- Google Serper API key

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/essay_writer.git
   cd PROJECT_DIRECTORY
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API key
   ```
   SERPER_API_KEY=your_api_key_here
   ```

### Usage

Run the example:

```bash
python main.py
```

To use in your own application:

```python
from main import graph

# Initialize the essay writer with your task
results = graph.invoke({
    'task': "The impact of artificial intelligence on healthcare",
    "max_revisions": 3,
    "revision_number": 1,
})

# Access the final essay
print(results["draft"])
```

## 📦 Dependencies

- **LangGraph**: Workflow orchestration
- **LangChain**: LLM integration framework
- **Ollama**: Local LLM inference
- **Google Serper**: Web search API

---

Made with ❤️ using LangGraph and LangChain 