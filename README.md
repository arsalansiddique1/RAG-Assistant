# RAG Assistant

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd rag-assistant
   ```

2. **Install [Poetry](https://python-poetry.org/)** (if you don't have it already):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**

   ```bash
   poetry install
   ```

4. **Configure environment variables**

   * Create a `.env` file in the project root:

     ```ini
     OPENAI_API_KEY=<your_openai_api_key>
     FLASK_SECRET_KEY=<a_random_session_secret>
     ```
   * Poetry will automatically load these when you run commands via `poetry run`.

5. **Run the application**

directly with Python

    ```bash
    poetry run python main.py
    ```

The app will be available at `http://localhost:5000`.

## Implementation Approach

* **Architecture**: A Flask backend exposes three main endpoints:

  1. `/upload` for PDF ingestion and vector store indexing
  2. `/chat` for RAG-based Q\&A
  3. **(Optional)** SSE streaming via the same `/chat` endpoint for live typing effect
* **RAG Chain**: Built with LangChain v0.3+

  * *History-aware retrieval*: Condenses follow-up questions into standalone queries
  * *Retrieval*: Hits a vector store (MMR search) to fetch relevant chunks
  * *Answering*: Stuffs retrieved context into a ChatOpenAI call
* **Front-end**: HTMX + Tailwind

  * PDF upload widget with client-side validation (extension + size)
  * Language selector drives prompt injection
  * Chat UI with user/assistant bubbles
  * SSE-based pseudo-streaming of completed answers for a typewriter effect
* **Session Memory**: Stored in Flask’s signed cookie, capped to the last 10 messages to avoid exceeding size limits

## Challenges Faced and Solutions

* **Deprecations in LangChain**: Migrated from `ConversationalRetrievalChain` & `ConversationBufferMemory` to LCEL primitives (`create_history_aware_retriever`, `create_retrieval_chain`).
* **Session Cookie Size**: Chat history in cookie exceeded 4 KB. Addressed by trimming to the last 20 messages before re-saving.


## Future Improvements

With more time, one could:

* **True Token-Level Streaming**: Perform retrieval up front and then call `llm.stream(...)` over the combined context for genuine low-latency output
* **Server-Side Session Store**: Migrate to Redis or filesystem sessions (via Flask-Session) to support longer histories and multi-user
* **Authentication & Multi-User**: Add user login, per-user history, and access controls
* **Error Handling & Monitoring**: Graceful fallback for API errors, usage metrics, and logging
* **UI Enhancements**: Markdown rendering, code snippet highlighting, conversation export, and a clear-history button
