import json
import requests
import time
import os
import re
import nltk
import asyncio
from typing import List, Dict, Any, Optional, Type
from functools import partial

# --- LangChain Core Imports ---
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)

# --- Other Libraries ---
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
# PLEASE REPLACE WITH YOUR OWN GEMINI API KEY
# You can get one from Google AI Studio.
API_KEY = ""
# LangChain uses this environment variable
os.environ["GOOGLE_API_KEY"] = API_KEY

# --- 1. Load Input Data (No changes) ---

def load_inputs(genomics_file, nadi_file):
    """Loads the genomics and nadi JSON files."""
    try:
        with open(genomics_file, 'r') as f:
            genomics_data = json.load(f)
        with open(nadi_file, 'r') as f:
            nadi_data = json.load(f)
        return genomics_data, nadi_data
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure '{genomics_file}' and '{nadi_file}' are in the same directory.")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None, None

# --- 2. Data Collection (Helper for Retriever) ---
# This function is now called *by* our LangChain retriever

def collect_journal_data(search_query, retmax=40):
    """
    Collects journal abstracts from PubMed E-utilities API.
    """
    print(f"Collecting data from PubMed for query: {search_query}")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    search_url = f"{base_url}esearch.fcgi"
    search_params = {
        "db": "pubmed", "term": search_query, "retmax": retmax, "retmode": "json"
    }
    
    try:
        response = requests.get(search_url, params=search_params)
        response.raise_for_status()
        search_data = response.json()
        
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            print("No articles found on PubMed for this query.")
            return []
            
        print(f"Found {len(id_list)} article IDs.")
        
        # Adding a small delay to be polite to the API
        time.sleep(0.5)
        
        fetch_url = f"{base_url}efetch.fcgi"
        fetch_params = {
            "db": "pubmed", "id": ",".join(id_list), "rettype": "abstract", "retmode": "text"
        }
        
        response = requests.get(fetch_url, params=fetch_params)
        response.raise_for_status()
        
        raw_text = response.text
        articles = []
        pmid_matches = list(re.finditer(r'PMID: (\d+)', raw_text))
        
        for i, match in enumerate(pmid_matches):
            pmid = match.group(1)
            start_index = match.end()
            end_index = pmid_matches[i+1].start() if i + 1 < len(pmid_matches) else len(raw_text)
            abstract_text = raw_text[start_index:end_index].strip()
            abstract_text = re.sub(r'\n\s*\n', '\n', abstract_text).replace("Abstract", "").strip()
            
            if abstract_text:
                articles.append({
                    "id": pmid,
                    "text": abstract_text,
                    "source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
        
        print(f"Successfully fetched {len(articles)} abstracts.\n")
        return articles

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from PubMed: {e}")
        return []

# --- 3. LangChain Custom Retriever (Unchanged) ---

class PubMedTfidfRetriever(BaseRetriever):
    """
    Custom LangChain Retriever that:
    1. Fetches articles from PubMed using `collect_journal_data`.
    2. Uses TF-IDF to find the most relevant sentences from those articles.
    """
    vectorizer: Any = Field(default_factory=TfidfVectorizer)
    top_k: int = 10
    
    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementation of the retriever."""
        
        # 1. Collect data from PubMed
        articles = collect_journal_data(query, retmax=40) # Use the larger retmax
        if not articles:
            return []

        # 2. Use TF-IDF logic
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK 'punkt' tokenizer...")
            nltk.download('punkt')
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Downloading NLTK 'punkt_tab' dependency...")
            nltk.download('punkt_tab')


        chunks = []
        chunk_metadata = []
        for article in articles:
            for sentence in sent_tokenize(article["text"]):
                chunks.append(sentence)
                chunk_metadata.append({
                    "id": article["id"],
                    "source": article["source"]
                })
        
        if not chunks:
            print("No text chunks found in articles.")
            return []

        try:
            self.vectorizer = TfidfVectorizer(stop_words='english')
            doc_vectors = self.vectorizer.fit_transform(chunks)
        except ValueError as e:
            print(f"Error in vectorizer (likely empty documents): {e}")
            return []
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        
        top_indices = similarities.argsort()[-self.top_k:][::-1]
        
        relevant_docs = []
        seen_ids = set()
        for i in top_indices:
            if similarities[i] > 0:
                metadata = chunk_metadata[i]
                pmid = metadata["id"]
                if pmid not in seen_ids:
                    doc = Document(
                        page_content=chunks[i],
                        metadata={"id": pmid, "source": metadata["source"]}
                    )
                    relevant_docs.append(doc)
                    seen_ids.add(pmid)
        
        print(f"Retrieved {len(relevant_docs)} relevant context snippets via TF-IDF.\n")
        return relevant_docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Async wrapper for the sync implementation."""
        loop = asyncio.get_event_loop()
        sync_run_manager = run_manager.get_sync()
        func_to_run = partial(
            self._get_relevant_documents, query, run_manager=sync_run_manager
        )
        return await loop.run_in_executor(None, func_to_run)

# --- 4. Define Structured Outputs (Expanded) ---

class FoodItem(BaseModel):
    """A single food recommendation."""
    food_item: str = Field(description="The name of the recommended food item.")
    quantity: str = Field(description="The recommended quantity or serving size (e.g., '1 cup daily', '100g 3 times/week').")
    reason_with_source: str = Field(description="The reason for the recommendation, MUST cite the source (e.g., 'Rich in fiber, (PMID: 123456)' or 'Improves digestion, (AyurCentral.com)').")

class FoodRecommendations(BaseModel):
    """The final JSON object containing a list of 5 food recommendations."""
    recommendations: List[FoodItem] = Field(description="A list of exactly 5 food items.", max_items=5, min_items=5)

class ExerciseItem(BaseModel):
    """A single exercise recommendation."""
    exercise_name: str = Field(description="The name of the recommended exercise or yoga asana (e.g., 'Brisk Walking', 'Surya Namaskar').")
    duration_frequency: str = Field(description="The recommended duration and frequency (e.g., '30 minutes, 5 days/week', '10 rounds daily').")
    reason_with_source: str = Field(description="The reason for the recommendation, MUST cite the source (e.g., 'Improves cardiovascular health, (PMID: 123456)' or 'Calms the mind, (YogaJournal.com)').")

class ExerciseRecommendations(BaseModel):
    """The final JSON object containing a list of 5 exercise recommendations."""
    recommendations: List[ExerciseItem] = Field(description="A list of exactly 5 exercise/yoga recommendations.", max_items=5, min_items=5)

# --- 5. Helper Functions for RAG Chains ---

def format_context(docs: List[Document]) -> str:
    """Helper function to format the retrieved documents for PubMed RAG."""
    if not docs:
        return "No relevant context found."
    return "\n\n".join(
        f"CONTEXT FROM PMID {doc.metadata['id']}:\n{doc.page_content}\nSource: {doc.metadata['source']}"
        for doc in docs
    )

async def run_rag_chain(search_query, user_llm_query, system_prompt, structured_llm, retriever, output_filename):
    """
    Runs the PubMed RAG pipeline for a given query and saves the JSON output.
    """
    print(f"\n--- Invoking PubMed RAG chain for {output_filename} ---")
    
    # 1. Define Prompt Template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "USER QUERY: {user_query}\n\nRELEVANT CONTEXT FROM PUBMED:\n{context}\n\nPlease generate the JSON array of top 5 recommendations.")
    ])
    
    # 2. Define Retriever Chain
    retriever_chain = (
        (lambda x: x["search_query"])
        | retriever
        | format_context
    )
    
    # 3. Define Full RAG Chain
    full_rag_chain = (
        RunnablePassthrough.assign(
            context=retriever_chain
        )
        | prompt_template
        | structured_llm
    )
    
    # 4. Define Input
    chain_input = {
        "search_query": search_query,
        "user_query": user_llm_query
    }
    
    # 5. Invoke Chain and Save Output
    try:
        recommendations_model = await full_rag_chain.ainvoke(chain_input)
        recommendations = recommendations_model.model_dump()
        
    except Exception as e:
        print(f"Error invoking LangChain RAG chain for {output_filename}: {e}")
        recommendations = {
            "error": f"Failed to generate recommendations for {output_filename}.",
            "details": str(e)
        }

    # 6. Save output
    with open(output_filename, 'w') as f:
        json.dump(recommendations, f, indent=2)
        
    print("="*50)
    print(f"Success! Recommendations saved to {output_filename}")
    print(json.dumps(recommendations, indent=2))
    print("="*50)

async def run_grounded_rag_chain(user_llm_query, system_prompt, output_schema: Type[BaseModel], output_filename):
    """
    Runs a Google-Search-Grounded RAG pipeline for a given query.
    """
    print(f"\n--- Invoking Google Search RAG chain for {output_filename} ---")
    
    try:
        # 1. Define Grounded LLM
        # This LLM is configured to use Google Search
        llm_grounded = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-09-2025",
            temperature=0.3,
            tools=[{"google_search": {}}] # This enables Google Search
        )
        
        # 2. Create the structured output version
        structured_llm_grounded = llm_grounded.with_structured_output(output_schema)
        
        # 3. Define Prompt Template (simpler, no context)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "USER QUERY: {user_query}\n\nPlease generate the JSON array of top 5 recommendations based on your knowledge and web search results.")
        ])
        
        # 4. Define Full RAG Chain (simpler, no retriever)
        full_rag_chain = prompt_template | structured_llm_grounded
        
        # 5. Define Input
        chain_input = {"user_query": user_llm_query}

        # 6. Invoke Chain and Save Output
        recommendations_model = await full_rag_chain.ainvoke(chain_input)
        recommendations = recommendations_model.model_dump()
        
    except Exception as e:
        print(f"Error invoking LangChain Grounded RAG chain for {output_filename}: {e}")
        recommendations = {
            "error": f"Failed to generate recommendations for {output_filename}.",
            "details": str(e)
        }

    # 7. Save output
    with open(output_filename, 'w') as f:
        json.dump(recommendations, f, indent=2)
        
    print("="*50)
    print(f"Success! Recommendations saved to {output_filename}")
    print(json.dumps(recommendations, indent=2))
    print("="*50)


# --- 6. Main Execution (Refactored for 2 Pipelines) ---

async def main():
    # Check for API Key
    if API_KEY == "YOUR_GEMINI_API_KEY":
        print("="*50)
        print("ERROR: Please update 'API_KEY' in food_recommender.py")
        print("You can get a key from Google AI Studio.")
        print("="*50)
        return

    # 1. Load data
    genomics_data, nadi_data = load_inputs("genomics.json", "nadi.json")
    if not genomics_data:
        return
        
    # 2. Define base patient info
    high_risk_diseases = [
        d["name"] for d in genomics_data.get("diseases", []) if d.get("z-score", 0) > 1.0
    ]
    disease_str = " AND ".join(high_risk_diseases) if high_risk_diseases else "health"
    nadi_str = nadi_data.get("parameters", {}).get("observed_characteristics", "unknown")
    
    # 3. Define the 4 search queries for PubMed
    # We only need the allopathic ones for the PubMed retriever
    query_allo_food = f"({disease_str}) AND (diet OR nutrition OR food) AND (recommendations OR guidelines)"
    query_allo_exercise = f"({disease_str}) AND (exercise OR physical activity OR rehabilitation) AND (recommendations OR guidelines)"
    
    # 4. Define the 4 user-facing LLM queries
    user_llm_query_food = (
        f"Based on a patient with high genetic risk for {disease_str} "
        f"and Nadi findings of '{nadi_str}', "
        "what are the top 5 food recommendations?"
    )
    user_llm_query_exercise = (
        f"Based on a patient with high genetic risk for {disease_str} "
        f"and Nadi findings of '{nadi_str}', "
        "what are the top 5 exercise or yoga recommendations?"
    )
    # We will add a specifier for the Ayurvedic queries to help the search
    user_llm_query_ayur_food = (
        f"Based on a patient with high genetic risk for {disease_str} "
        f"and Nadi findings of '{nadi_str}', "
        "what are the top 5 *Ayurvedic* food or herb recommendations?"
    )
    user_llm_query_ayur_yoga = (
        f"Based on a patient with high genetic risk for {disease_str} "
        f"and Nadi findings of '{nadi_str}', "
        "what are the top 5 *Yoga Asana or Pranayama* recommendations for balancing their condition?"
    )
    
    # 5. Define the 4 system prompts
    prompt_allo_food = (
        "You are a medical and nutritional expert. Your task is to answer the user's query "
        "based *only* on the provided context from allopathic journal abstracts. "
        "You must provide a final answer as a JSON array of 5 food items. "
        "For each item, provide a recommended quantity and a reason. "
        "The 'reason_with_source' MUST cite the PubMed ID (PMID) from the context. "
        "You must provide a specific quantity. If the context does not specify a quantity, use your expert knowledge to suggest a reasonable one (e.g., '1 cup daily')."
    )
    
    prompt_allo_exercise = (
        "You are a medical doctor and physical therapy expert. Your task is to answer the user's query "
        "based *only* on the provided context from allopathic journal abstracts. "
        "You must provide a final answer as a JSON array of 5 exercise recommendations. "
        "For each item, provide a recommended duration/frequency (e.g., '30 minutes, 3 times/week'). "
        "The 'reason_with_source' MUST cite the PubMed ID (PMID) from the context. "
        "If the context does not specify a duration, use your expert knowledge to suggest a reasonable one."
    )
    
    prompt_ayur_food = (
        "You are an expert Ayurvedic practitioner (Vaidya). Your task is to answer the user's query. "
        "Use your internal knowledge and Google Search to find relevant Ayurvedic information. "
        "You must provide a final answer as a JSON array of 5 Ayurvedic food or herb recommendations. "
        "For each item, provide a recommended quantity (e.g., '1 tsp with warm water'). "
        "The 'reason_with_source' MUST cite the web source (e.g., 'BanyanBotanicals.com', 'Journal of Ayurveda'). "
        "If no quantity is found, use your expert Ayurvedic knowledge to suggest a reasonable one."
    )
    
    prompt_ayur_yoga = (
        "You are an expert Ayurvedic practitioner and Yoga Acharya. Your task is to answer the user's query. "
        "Use your internal knowledge and Google Search to find relevant Ayurvedic and Yogic information. "
        "You must provide a final answer as a JSON array of 5 Yoga asana or Pranayama recommendations. "
        "For each item, provide a recommended duration/frequency (e.g., '5 rounds daily', 'Hold for 30 seconds'). "
        "The 'reason_with_source' MUST cite the web source (e.g., 'YogaJournal.com', 'ArtOfLiving.org'). "
        "If no duration is found, use your expert knowledge to suggest a reasonable one."
    )

    # 6. Define LangChain Components for PubMed (Allopathic) RAG
    # This is the non-grounded LLM
    llm_pubmed = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0.3)
    food_llm = llm_pubmed.with_structured_output(FoodRecommendations)
    exercise_llm = llm_pubmed.with_structured_output(ExerciseRecommendations)
    
    retriever = PubMedTfidfRetriever(top_k=15)
    
    # 7. Run the 4 chains in sequence
    
    # --- Allopathic Pipeline ---
    await run_rag_chain(
        search_query=query_allo_food,
        user_llm_query=user_llm_query_food,
        system_prompt=prompt_allo_food,
        structured_llm=food_llm,
        retriever=retriever,
        output_filename="food_allopathic.json"
    )
    
    await run_rag_chain(
        search_query=query_allo_exercise,
        user_llm_query=user_llm_query_exercise,
        system_prompt=prompt_allo_exercise,
        structured_llm=exercise_llm,
        retriever=retriever,
        output_filename="exercise_allopathic.json"
    )
    
    # --- Ayurvedic / Grounded Pipeline ---
    await run_grounded_rag_chain(
        user_llm_query=user_llm_query_ayur_food,
        system_prompt=prompt_ayur_food,
        output_schema=FoodRecommendations, # Pass the Pydantic class
        output_filename="food_ayurvedic.json"
    )
    
    await run_grounded_rag_chain(
        user_llm_query=user_llm_query_ayur_yoga,
        system_prompt=prompt_ayur_yoga,
        output_schema=ExerciseRecommendations, # Pass the Pydantic class
        output_filename="yoga_ayurvedic.json"
    )
    
    print("\nAll 4 recommendation files have been generated.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "cannot run nested" in str(e):
            print("Running in a nested asyncio environment. Using 'await'.")
        else:
            raise e

