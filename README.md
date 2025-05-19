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

This implementation's main focus was to offer speed of execution and simplicity of development as mentioned in the spec. Hence, only some advanced RAG techniques that offer better retrieval accuracy were employed. If the task was for a niche use case, retrieval performance could be significantly improved at the cost of latency.

Code meets all requirements stated in spec, including all HTMX enchancements.

* **Architecture**: A Flask backend exposes three main endpoints:

  1. `/upload` for PDF ingestion and vector store indexing
  2. `/chat` for RAG-based Q\&A
  3. `/clear` additional feature for clearing session history, uploads and vector db if page is closed
* **RAG Chain**: Built with LangChain v0.3+

  * *History-aware retrieval*: Condenses follow-up questions into standalone queries
  * *Retrieval*: Uses Qdrant hybrd-search with BM25 and text-embedding-003 large for sparse/dense vectors, RRF for fusing rankings.
  * *Answering*: Stuffs retrieved context into a ChatOpenAI call
* **Front-end**: HTMX + Tailwind

  * PDF upload widget with client-side validation (extension + size)
  * Language selector drives prompt injection
  * Chat UI with user/assistant bubbles and typewriter streaming effect
   
* **Session Memory**: Stored in Flask’s signed cookie, capped to the last 20 messages to avoid exceeding size limits

## Challenges Faced and Solutions

* **Deprecations in LangChain**: Migrated from `ConversationalRetrievalChain` & `ConversationBufferMemory` to LCEL primitives (`create_history_aware_retriever`, `create_retrieval_chain`).
* **Session Cookie Size**: Chat history in cookie exceeded 4 KB. Addressed by trimming to the last 20 messages before re-saving.


## Future Improvements

With more time, one could:

* **Data Ingestion**: Improve data ingestion capability by detecting if pdf is scanned and having OCR model to detect text layer when processing docs.
* **Chunking**: Section-wise chunking of structured documents would significantly improve retrieval accuracy. This can be done by using aryn API or alternatively converting to markdown via docling, then identifying section start line numbers through llm calls.
* **Retrieval**: Can be improved vastly by carrying out a multi-stage hybrid search. i.e. generate metadata summaries for each chunk using LLM calls, store in Qdrant. For any new query, search through chunk summaries then use chunk summaries to point to relevant chunks.
* **Sources/citations**: By using section-wise chunking, chunks can have header metadata which provides useful context for the user to know exactly where in the document the information was extracted from.
* **True Token-Level Streaming**: Perform retrieval up front and then call `llm.stream(...)` over the combined context for genuine low-latency output
* **Server-Side Session Store**: Migrate to Redis or filesystem sessions (via Flask-Session) to support longer histories and multi-user
* **Authentication & Multi-User**: Add user login, per-user history, and access controls
* **Error Handling & Monitoring**: Graceful fallback for API errors, usage metrics, and logging
* **UI Enhancements**: Markdown rendering, code snippet highlighting, conversation export, and a clear-history button
* **Latency vs Performance tradeoff**: Adding in contextual chunking will increase latency however will significantly improve RAG retrieval accuracy. Useful for when RAG is for a highly-complex niche task that requires high accuracy. The point mentioned above for retrieval would also signifcantly improve RAG accuracy for niche tasks. These improvements work well in cases where latency isn't the main priority.

