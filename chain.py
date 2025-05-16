from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

def get_qa_chain(llm, retriever):
    """
    Build a history-aware conversational retrieval chain using LangChain 0.3+ LCEL.
    """

    # 1. Prompt to condense follow-up questions into standalone questions
    condense_question_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "Given the following conversation and a follow up question, "
            "rephrase the follow up question to be a standalone question. "
            "If it is already standalone, return it unchanged."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])

    # 2. Wrap the retriever to handle history and follow-up rephrasing
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=condense_question_prompt,
    )

    # 3. Prompt to answer the question using retrieved context
    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say so. Keep responses concise.\n\n{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])
    qa_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
    )

    # 4. Combine retriever and QA chain into a single LCEL chain
    return create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=qa_chain,
    )
