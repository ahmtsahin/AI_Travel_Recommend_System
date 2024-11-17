from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
import spacy

def setup_chatbot(llm, vector_db):
    """Setup the chatbot with LLM and vector database."""
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    template = """You are a friendly travel recommendation chatbot. Answer the question based on the following context, previous conversation, and the information from the dataframe. 
    If asked to inspire with travel destinations, suggest cities from the dataframe. When a user expresses interest in a specific city, provide relevant information including attractions, images, and links about this city.
    Only ask about hotel preferences (like number of rooms and budget) if the user specifically requests hotel recommendations.

    Previous conversation: {chat_history}
    Context to answer question: {context}
    New human question: {question}
    Response:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question", "chat_history"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return chain

def extract_city_nlp(user_input, nlp):
    """Extract city name from user input using spaCy."""
    doc = nlp(user_input)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text
    return "City not found"