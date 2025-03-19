# Agentic GenAI App

Agentic GenAI App is a Streamlit-based application that leverages multiple AI agents to perform tasks such as web searches, financial data analysis, and knowledge-based queries. The app integrates Groq-powered models, custom embeddings, and tools like DuckDuckGo and YFinance to provide intelligent responses to user queries.

---

## Table of Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Key Components](#key-components)
6. [Dependencies](#dependencies)
7. [Environment Variables](#environment-variables)
8. [Customization](#customization)
9. [License](#license)
10. [Acknowledgments](#acknowledgments)
11. [Contact](#contact)

---

## Features

- **Web Agent**: Searches the web for information using DuckDuckGo and includes sources in the response.
- **Finance Agent**: Retrieves financial data, including stock prices, analyst recommendations, and company fundamentals, and displays them in tables.
- **Chef Agent**: Uses a PDF knowledge base (e.g., Thai recipes) with a custom HuggingFace embedder to answer culinary questions.
- **Team Agent**: Combines the Web and Finance agents to provide comprehensive responses.
- **Conversation Memory**: Maintains conversation history for each agent, allowing for contextual responses.
- **Sidebar Chat History**: Displays the conversation history in the sidebar with an option to clear it.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/Agentic-GenAI-App.git
   cd Agentic-GenAI-App
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Create a `.env` file in the root directory.
   - Add the following keys:
     ```
     GROQ_API_KEY=your_groq_api_key
     ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. Open the app in your browser (default: [http://localhost:8501](http://localhost:8501)).
2. Select an agent from the dropdown menu:
   - Web Agent
   - Finance Agent
   - Chef Agent
   - Team Agent
3. Enter your query in the text input field.
4. Click the "Submit Query" button to get a response.
5. View the conversation history in the sidebar.

---

## Project Structure

```
Agentic-GenAI-App/
├── agents/
│   ├── web_agent.py
│   ├── finance_agent.py
│   ├── chef_agent.py
│   └── team_agent.py
├── knowledge_base/
│   ├── pdf_knowledge_base.py
│   └── lance_db/
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

## Key Components

### Agents
- **Web Agent**: Uses DuckDuckGo for web searches.
- **Finance Agent**: Retrieves financial data using YFinance.
- **Chef Agent**: Answers culinary questions using a PDF knowledge base and a custom HuggingFace embedder.
- **Team Agent**: Combines Web and Finance agents for multi-agent collaboration.

### Knowledge Base
- The Chef Agent uses a PDF knowledge base stored in LanceDB for answering domain-specific questions.

### Conversation Memory
- The app maintains conversation history for each agent using Streamlit's session state, enabling contextual responses.

---

## Dependencies

- Python 3.8+
- Streamlit
- HuggingFace Transformers
- Groq
- LanceDB
- YFinance
- DuckDuckGo-Search
- Python-Dotenv

---

## Environment Variables

The app requires the following environment variables to be set in a `.env` file:

- `GROQ_API_KEY`: API key for Groq models.

---

## Customization

1. **Custom Embedder**:
   - The app uses a HuggingFace embedder for the Chef Agent.
   - You can modify the model by changing the `model_name` in the `HuggingFaceEmbedder` class.

2. **Knowledge Base**:
   - Update the `urls` in the `PDFUrlKnowledgeBase` to use a different PDF for the Chef Agent.

---



---

## Acknowledgments

- **Streamlit** for the interactive UI.
- **HuggingFace** for the Transformers library.
- **DuckDuckGo** for web search integration.
- **YFinance** for financial data retrieval.
- **Groq** for AI model support.

---

## Contact

For questions or feedback, please contact the owner of this repo or create an issue.
