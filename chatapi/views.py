from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .utils import load_and_index_docs
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from datetime import datetime

load_dotenv()

# knowledge base
vectorstore = load_and_index_docs()

# initialize Groq llm
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

custom_prompt =  PromptTemplate.from_template(
    """You represent our company. Use "we","our" and "us". If you don't know, say you don't have the information. 

Context: {context}

Question: {question}

Answer:"""
)



# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}),
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=False
)


class ChatView(APIView):
    def post(self, request):
        question = request.data.get("question", "").strip()
        if not question:
            return Response({"error": "Missing 'question' field in request."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            response = qa_chain.invoke(question)
            if isinstance(response, dict):
                response_text = response.get("result", "")
            else:
                response_text = response
        
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

            with open("chat_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"{timestamp}\n")
                log_file.write(f"Question: {question}\n")
                log_file.write(f"Response: {response_text}\n")
                
                log_file.write("------------------------------------------------------------\n")

            return Response({"response": response})
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)