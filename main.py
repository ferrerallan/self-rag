import openai
import faiss
import numpy as np
from typing import List, Dict
import json
import os
from dotenv import load_dotenv

load_dotenv()

class SelfRAG:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.dimension = 1536
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []

    def add_documents(self, docs: List[str]):
        embeddings = []
        for doc in docs:
            response = self.client.embeddings.create(
                input=doc,
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
            self.documents.append(doc)
        
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)

    def generate_response(self, question: str, context: List[str]) -> str:
        prompt = f"""
        Context: {' '.join(context)}
        Question: {question}
        Generate a response based only on the provided context.
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that answers based only on the provided context."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def evaluate_response(self, question: str, response: str, context: List[str]) -> Dict:
        eval_prompt = f"""
        Evaluate this response:
        Question: {question}
        Context: {context}
        Response: {response}
        
        Rate (1-10):
        1. Relevance to question
        2. Use of context
        3. Accuracy
        
        Return JSON format: {{"scores": {{"relevance": X, "context_usage": X, "accuracy": X}}, "suggestions": "improvement notes"}}
        """
        
        evaluation = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a critical evaluator."}, 
                {"role": "user", "content": eval_prompt}
            ]
        )
        return json.loads(evaluation.choices[0].message.content)

    def refine_response(self, question: str, context: List[str], original_response: str, evaluation: Dict) -> str:
        if sum(evaluation['scores'].values()) / 3 >= 9.3:
            return original_response
            
        refine_prompt = f"""
        Original question: {question}
        Context: {context}
        Original response: {original_response}
        Evaluation: {evaluation}
        
        Provide an improved response addressing the suggestions.
        """
        
        refined = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Improve the response based on evaluation."},
                {"role": "user", "content": refine_prompt}
            ]
        )
        return refined.choices[0].message.content

    def query(self, question: str, k: int = 3) -> Dict:
        question_embedding = self.client.embeddings.create(
            input=question,
            model="text-embedding-ada-002"
        ).data[0].embedding

        D, I = self.index.search(np.array([question_embedding]).astype('float32'), k)
        context = [self.documents[i] for i in I[0]]
        
        initial_response = self.generate_response(question, context)
        evaluation = self.evaluate_response(question, initial_response, context)
        final_response = self.refine_response(question, context, initial_response, evaluation)
        
        return {
            "context": context,
            "initial_response": initial_response,
            "evaluation": evaluation,
            "final_response": final_response
        }

if __name__ == "__main__":
    api = os.getenv("OPENAI_KEY")
    rag = SelfRAG(api)
    
    docs = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "FAISS is a library developed by Facebook for efficient similarity search and clustering of dense vectors.",
        "OpenAI develops advanced language models and conducts research in artificial intelligence.",
        "Python was created by Guido van Rossum and was first released in 1991.",
        "Machine learning algorithms require significant computational resources and large datasets."
    ]
    rag.add_documents(docs)
    
    question = "Tell me about Python?"
    result = rag.query(question)
    
    print("\nContext used:", result["context"])
    print("\nInitial response:", result["initial_response"])
    print("\nEvaluation:", result["evaluation"])
    print("\nFinal response:", result["final_response"])