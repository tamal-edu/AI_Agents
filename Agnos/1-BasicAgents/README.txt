Agentic GenAI App
Agentic GenAI App is a Streamlit-based application that leverages multiple AI agents to perform tasks such as web searches, financial data analysis, and knowledge-based queries. The app integrates Groq-powered models, custom embeddings, and tools like DuckDuckGo and YFinance to provide intelligent responses to user queries.

Features
Web Agent: Searches the web for information using DuckDuckGo and includes sources in the response.
Finance Agent: Retrieves financial data, including stock prices, analyst recommendations, and company fundamentals, and displays them in tables.
Chef Agent: Uses a PDF knowledge base (e.g., Thai recipes) with a custom HuggingFace embedder to answer culinary questions.
Team Agent: Combines the Web and Finance agents to provide comprehensive responses.
Conversation Memory: Maintains conversation history for each agent, allowing for contextual responses.
Sidebar Chat History: Displays the conversation history in the sidebar with an option to clear it.
Installation
Clone the repository:

Install dependencies:

Set up environment variables:

Create a .env file in the root directory.
Add the following keys:
Run the app:

Usage
Open the app in your browser (default: http://localhost:8501).
Select an agent from the dropdown menu:
Web Agent
Finance Agent
Chef Agent
Team Agent
Enter your query in the text input field.
Click the "Submit Query" button to get a response.
View the conversation history in the sidebar.
Project Structure
Key Components
Agents
Web Agent: Uses DuckDuckGo for web searches.
Finance Agent: Retrieves financial data using YFinance.
Chef Agent: Answers culinary questions using a PDF knowledge base and a custom HuggingFace embedder.
Team Agent: Combines Web and Finance agents for multi-agent collaboration.
Knowledge Base
The Chef Agent uses a PDF knowledge base stored in LanceDB for answering domain-specific questions.
Conversation Memory
The app maintains conversation history for each agent using Streamlit's session state, enabling contextual responses.
Dependencies
Python 3.8+
Streamlit
HuggingFace Transformers
Groq
LanceDB
YFinance
DuckDuckGo-Search
Python-Dotenv
Environment Variables
The app requires the following environment variables to be set in a .env file:

GROQ_API_KEY: API key for Groq models.
Customization
Custom Embedder: The app uses a HuggingFace embedder for the Chef Agent. You can modify the model by changing the model_name in the HuggingFaceEmbedder class.
Knowledge Base: Update the urls in the PDFUrlKnowledgeBase to use a different PDF for the Chef Agent.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Streamlit for the interactive UI.
HuggingFace for the Transformers library.
DuckDuckGo for web search integration.
YFinance for financial data retrieval.
Groq for AI model support.
Contact
For questions or feedback, please contact [Your Name] at [Your Email].## Key Components

Agents
Web Agent: Uses DuckDuckGo for web searches.
Finance Agent: Retrieves financial data using YFinance.
Chef Agent: Answers culinary questions using a PDF knowledge base and a custom HuggingFace embedder.
Team Agent: Combines Web and Finance agents for multi-agent collaboration.
Knowledge Base
The Chef Agent uses a PDF knowledge base stored in LanceDB for answering domain-specific questions.
Conversation Memory
The app maintains conversation history for each agent using Streamlit's session state, enabling contextual responses.
Dependencies
Python 3.8+
Streamlit
HuggingFace Transformers
Groq
LanceDB
YFinance
DuckDuckGo-Search
Python-Dotenv
Environment Variables
The app requires the following environment variables to be set in a .env file:

GROQ_API_KEY: API key for Groq models.
Customization
Custom Embedder: The app uses a HuggingFace embedder for the Chef Agent. You can modify the model by changing the model_name in the HuggingFaceEmbedder class.
Knowledge Base: Update the urls in the PDFUrlKnowledgeBase to use a different PDF for the Chef Agent.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Streamlit for the interactive UI.
HuggingFace for the Transformers library.
DuckDuckGo for web search integration.
YFinance for financial data retrieval.
Groq for AI model support.