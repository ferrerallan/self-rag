import openai
import faiss
import numpy as np
from typing import List, Dict, Tuple
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
        self.retrieval_threshold = 0.2  # Threshold to decide if retrieval is needed

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

    def decide_retrieval_need(self, question: str) -> bool:
        """Determines if retrieval is needed for this question"""
        prompt = f"""
        Instruction: {question}
        Determine if finding external documents would help answer this question better.
        Output only 'Yes' or 'No'.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You decide if retrieval is needed for a question."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        retrieve_decision = response.choices[0].message.content.strip().lower()
        return retrieve_decision == "yes"

    def evaluate_passage_relevance(self, question: str, passage: str) -> Tuple[str, float]:
        """Evaluates if a passage is relevant to the question"""
        prompt = f"""
        Question: {question}
        Passage: {passage}
        
        Is this passage relevant to answering the question?
        Output only 'Relevant' or 'Irrelevant'.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You evaluate passage relevance."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        relevance = response.choices[0].message.content.strip()
        score = 1.0 if relevance.lower() == "relevant" else 0.0
        return relevance, score

    def evaluate_support(self, question: str, response: str, passage: str) -> Tuple[str, float]:
        """Evaluates if the response is supported by the passage"""
        prompt = f"""
        Question: {question}
        Passage: {passage}
        Response: {response}
        
        To what extent is this response supported by the passage?
        Output one of: 'Fully supported', 'Partially supported', 'No support'.
        """
        
        evaluation = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You evaluate factual support."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        support_level = evaluation.choices[0].message.content.strip()
        
        if "fully" in support_level.lower():
            score = 1.0
        elif "partially" in support_level.lower():
            score = 0.5
        else:
            score = 0.0
            
        return support_level, score

    def evaluate_usefulness(self, question: str, response: str) -> Tuple[int, float]:
        """Evaluates the overall usefulness of the response"""
        prompt = f"""
        Question: {question}
        Response: {response}
        
        Rate the usefulness of this response on a scale of 1-5, where 5 is most useful.
        Output only the number.
        """
        
        evaluation = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You evaluate response usefulness."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        try:
            usefulness = int(evaluation.choices[0].message.content.strip())
        except:
            usefulness = 3  # Default if parsing fails
            
        normalized_score = usefulness / 5.0
        return usefulness, normalized_score

    def generate_response(self, question: str, passages: List[str]) -> str:
        """Generates a response using adaptive retrieval"""
        # Create a context from combined passages
        context = "\n\n".join([f"Passage {i+1}: {passage}" for i, passage in enumerate(passages)])
        
        prompt = f"""
        Question: {question}
        
        Context:
        {context}
        
        Answer the question based on the provided context. If the context doesn't contain relevant information, 
        mention that fact in your response.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content

    def query(self, question: str, k: int = 5) -> Dict:
        """Main query method implementing SELF-RAG approach"""
        results = {
            "question": question,
            "retrieval_needed": False,
            "passages": [],
            "response": "",
            "reflection_tokens": {}
        }
        
        # Step 1: Decide if retrieval is needed
        retrieve_needed = self.decide_retrieval_need(question)
        results["retrieval_needed"] = retrieve_needed
        
        if not retrieve_needed:
            # Generate response without retrieval
            direct_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ]
            ).choices[0].message.content
            
            results["response"] = direct_response
            usefulness, usefulness_score = self.evaluate_usefulness(question, direct_response)
            results["reflection_tokens"]["usefulness"] = usefulness
            
            return results
        
        # Step 2: Retrieve passages
        question_embedding = self.client.embeddings.create(
            input=question,
            model="text-embedding-ada-002"
        ).data[0].embedding

        D, I = self.index.search(np.array([question_embedding]).astype('float32'), k)
        passages = [self.documents[i] for i in I[0]]
        
        # Step 3: Evaluate passage relevance
        relevant_passages = []
        relevance_results = []
        
        for passage in passages:
            relevance, relevance_score = self.evaluate_passage_relevance(question, passage)
            if relevance.lower() == "relevant":
                relevant_passages.append(passage)
            
            relevance_results.append({
                "passage": passage,
                "relevance": relevance,
                "relevance_score": relevance_score
            })
        
        results["passages"] = relevance_results
        
        # If no relevant passages found
        if not relevant_passages:
            direct_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ]
            ).choices[0].message.content
            
            results["response"] = direct_response
            usefulness, usefulness_score = self.evaluate_usefulness(question, direct_response)
            results["reflection_tokens"]["usefulness"] = usefulness
            
            return results
        
        # Step 4: Generate response with relevant passages
        response = self.generate_response(question, relevant_passages)
        
        # Step 5: Evaluate support for response from passages
        support_results = []
        for passage in relevant_passages:
            support_level, support_score = self.evaluate_support(question, response, passage)
            support_results.append({
                "passage": passage,
                "support_level": support_level,
                "support_score": support_score
            })
        
        # Step 6: Evaluate overall usefulness
        usefulness, usefulness_score = self.evaluate_usefulness(question, response)
        
        # Step 7: Compile results
        results["response"] = response
        results["reflection_tokens"] = {
            "support": support_results,
            "usefulness": usefulness
        }
        
        return results

if __name__ == "__main__":
    load_dotenv()
    api = os.getenv("OPENAI_API_KEY")
    print(api)
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
    
    print("\nQuestion:", result["question"])
    print("\nRetrieval needed:", result["retrieval_needed"])
    print("\nPassages:", result["passages"])
    print("\nResponse:", result["response"])
    print("\nReflection tokens:", result["reflection_tokens"])