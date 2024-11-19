from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import spacy

def setup_chatbot(llm, vector_db):
    """
    Set up the chatbot with the specified language model and vector database.
    
    Args:
        llm: Language model
        vector_db: Vector database for retrieval
        
    Returns:
        ConversationalRetrievalChain: Configured chatbot chain
    """
    try:
        # Set up memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create the chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_db.as_retriever(),
            memory=memory,
            return_source_documents=True
        )
        
        return chain
        
    except Exception as e:
        print(f"Error in setup_chatbot: {str(e)}")
        return None

def extract_city_nlp(text, nlp):
    """
    Extract city name from text using spaCy NLP.
    
    Args:
        text (str): Input text
        nlp: spaCy NLP model
        
    Returns:
        str: Extracted city name or empty string if not found
    """
    try:
        # Process the text
        doc = nlp(text)
        
        # Look for GPE (Geo-Political Entity) in the text
        for ent in doc.ents:
            if ent.label_ == 'GPE':
                return ent.text
        
        # If no city found, try to find proper nouns
        for token in doc:
            if token.pos_ == 'PROPN':
                return token.text
        
        return ""
        
    except Exception as e:
        print(f"Error in extract_city_nlp: {str(e)}")
        return ""
