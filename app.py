from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime
import uuid
from typing import List, Dict, Any
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)

# CORS Configuration - Allow all origins for maximum compatibility
CORS(app, 
     resources={r"/*": {
         "origins": "*",
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "X-Session-ID"],
         "expose_headers": ["Content-Type"],
         "supports_credentials": False
     }})

# Setup logging for message tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chat_logs.log'),
        logging.StreamHandler()
    ]
)

# Create logs directory if it doesn't exist
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)


# ==================== RAG SYSTEM CLASS (PURE GROQ - NO EMBEDDINGS) ====================
class RAGSystem:
    """
    Pure Groq RAG System - NO embeddings, NO vector databases, NO external dependencies.
    Uses simple keyword matching for document retrieval.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.documents = []
        self.company_summary = ""
        self.all_content = ""

    def load_company_data(self) -> Dict[str, Any]:
        """Load company data from data.json file."""
        data_file = Path(__file__).parent / "data.json"
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            raise FileNotFoundError("data.json file not found!")

    def _generate_summary_text(self, data: Dict[str, Any]) -> str:
        """Generates a high-level summary string from the company data."""
        company = data.get("company", {})
        services = data.get("services", [])
        why_choose_us = data.get("why_choose_us", [])
        boards = data.get("boards_supported", [])

        summary_parts = [
            f"Company Overview: {company.get('name', 'N/A')}",
            f"Location: {company.get('location', 'N/A')}",
            f"Established: {company.get('established', 'N/A')}",
            f"Description: {company.get('description', 'N/A')}",
            f"\nTotal Services Offered: {len(services)}",
            f"Educational Boards Supported: {', '.join(boards)}",
            f"\nKey Differentiators: {len(why_choose_us)} main reasons to choose us",
        ]

        if services:
            service_names = [s.get("name", "N/A") for s in services]
            summary_parts.append(f"Services include: {', '.join(service_names)}")

        return "\n".join(summary_parts)

    def _create_documents_from_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Creates a list of documents from the company data."""
        documents = []

        # Company Basic Info
        company = data.get("company", {})
        if company:
            company_content = f"""Company Name: {company.get('name', 'N/A')}
Location: {company.get('location', 'N/A')}
Established: {company.get('established', 'N/A')}
Website: {company.get('website', 'N/A')}
Email: {company.get('email', 'N/A')}
Description: {company.get('description', 'N/A')}"""
            documents.append({
                "content": company_content,
                "metadata": {"type": "company_info"},
                "keywords": ["company", "about", "contact", "email", "website", "location"]
            })

        # Services
        for service in data.get("services", []):
            service_content = f"""Service: {service.get('name', 'N/A')}
Description: {service.get('description', 'N/A')}"""
            service_name = service.get('name', '').lower()
            documents.append({
                "content": service_content,
                "metadata": {"type": "service", "service_name": service.get("name", "N/A")},
                "keywords": ["service", "offer", "provide"] + service_name.split()
            })

        # Why Choose Us
        for reason in data.get("why_choose_us", []):
            reason_content = f"""Reason: {reason.get('title', 'N/A')}
Details: {reason.get('description', 'N/A')}"""
            documents.append({
                "content": reason_content,
                "metadata": {"type": "why_choose_us", "title": reason.get("title", "N/A")},
                "keywords": ["why", "choose", "benefit", "advantage", "feature"]
            })

        # Demo Process
        demo = data.get("demo_process", {})
        if demo:
            steps = demo.get("steps", [])
            demo_content = f"""Demo Process:
Steps: {'; '.join(steps)}
Demo Fee: {demo.get('demo_fee', 'N/A')}
Profile Finalization Time: {demo.get('profile_finalization_time', 'N/A')}
Demo Duration: {demo.get('demo_duration', 'N/A')}"""
            documents.append({
                "content": demo_content,
                "metadata": {"type": "demo_process"},
                "keywords": ["demo", "trial", "test", "process", "how", "steps"]
            })

        # Parent Instructions
        parent_instructions = data.get("parent_instructions", [])
        if parent_instructions:
            documents.append({
                "content": f"Parent Instructions:\n" + "\n".join([f"- {inst}" for inst in parent_instructions]),
                "metadata": {"type": "parent_instructions"},
                "keywords": ["parent", "instruction", "guide", "how to", "steps"]
            })

        # Tutor Instructions
        tutor_instructions = data.get("tutor_instructions", [])
        if tutor_instructions:
            documents.append({
                "content": f"Tutor Instructions:\n" + "\n".join([f"- {inst}" for inst in tutor_instructions]),
                "metadata": {"type": "tutor_instructions"},
                "keywords": ["tutor", "teacher", "instruction", "guide"]
            })

        # Student Flow
        student_flow = data.get("student_flow", [])
        if student_flow:
            documents.append({
                "content": f"Student Flow:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(student_flow)]),
                "metadata": {"type": "student_flow"},
                "keywords": ["student", "flow", "process", "how", "enroll"]
            })

        # Tutor Flow
        tutor_flow = data.get("tutor_flow", [])
        if tutor_flow:
            documents.append({
                "content": f"Tutor Flow:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(tutor_flow)]),
                "metadata": {"type": "tutor_flow"},
                "keywords": ["tutor", "teacher", "flow", "process", "join"]
            })

        # Matching System
        matching = data.get("matching_system", {})
        if matching:
            factors = matching.get("factors", [])
            matching_content = f"""Matching System:
Type: {matching.get('type', 'N/A')}
Factors Considered: {', '.join(factors)}
Description: {matching.get('description', 'N/A')}"""
            documents.append({
                "content": matching_content,
                "metadata": {"type": "matching_system"},
                "keywords": ["match", "matching", "pair", "connect", "algorithm"]
            })

        # Payment Info
        payment = data.get("payment_info", {})
        if payment:
            payment_content = f"""Payment Information:
Payment Options: {payment.get('payment_options', 'N/A')}
Package Options: {payment.get('package_options', 'N/A')}
Demo Fee: {payment.get('demo_fee', 'N/A')}
Policy: {payment.get('policy', 'N/A')}"""
            documents.append({
                "content": payment_content,
                "metadata": {"type": "payment_info"},
                "keywords": ["payment", "fee", "cost", "price", "pay", "package"]
            })

        # Boards and Modes
        boards = data.get("boards_supported", [])
        modes = data.get("class_modes", [])
        if boards or modes:
            documents.append({
                "content": f"Educational Boards Supported: {', '.join(boards)}\nClass Modes Available: {', '.join(modes)}",
                "metadata": {"type": "boards_and_modes"},
                "keywords": ["board", "curriculum", "mode", "online", "offline"] + [b.lower() for b in boards]
            })

        # FAQs
        for faq in data.get("faq", []):
            faq_content = f"""Question: {faq.get('question', 'N/A')}
Answer: {faq.get('answer', 'N/A')}"""
            documents.append({
                "content": faq_content,
                "metadata": {"type": "faq", "question": faq.get("question", "N/A")},
                "keywords": ["faq", "question", "answer"] + faq.get('question', '').lower().split()
            })

        return documents

    def build_vectorstore(self):
        """Builds the document store - NO vectors, just text storage."""
        print("🔧 Building document store (Pure Groq - No Embeddings)...")

        data = self.load_company_data()

        # Generate and store the summary
        self.company_summary = self._generate_summary_text(data)
        print("✅ Company summary generated and cached.")

        # Create documents
        self.documents = self._create_documents_from_data(data)
        
        # Create a single concatenated content for fallback
        self.all_content = "\n\n".join([doc["content"] for doc in self.documents])
        
        print(f"📄 Created {len(self.documents)} documents.")
        print("✅ Document store built successfully! (No embeddings used)")

    def get_summary_document(self) -> str:
        """Returns the cached high-level summary text."""
        return self.company_summary

    def _clean_query(self, query: str) -> List[str]:
        """Clean and tokenize query into keywords."""
        # Remove punctuation and convert to lowercase
        query = re.sub(r'[^\w\s]', ' ', query.lower())
        # Split into words and remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'what', 'how', 'can', 'do', 'does'}
        words = [w for w in query.split() if w and w not in stop_words]
        return words

    def _calculate_relevance_score(self, query_words: List[str], document: Dict[str, Any]) -> float:
        """Calculate relevance score using keyword matching."""
        if not query_words:
            return 0.0
        
        # Check content
        content_lower = document["content"].lower()
        content_matches = sum(1 for word in query_words if word in content_lower)
        
        # Check keywords with higher weight
        keyword_matches = sum(1 for word in query_words if word in document.get("keywords", []))
        
        # Calculate weighted score
        total_score = (content_matches * 1.0) + (keyword_matches * 2.0)
        max_score = len(query_words) * 3.0
        
        return total_score / max_score if max_score > 0 else 0.0

    def search_relevant_context(self, query: str, k: int = 5) -> str:
        """Search for relevant context using keyword matching only."""
        if not self.documents:
            raise ValueError("Documents not loaded. Call build_vectorstore() first.")

        # Clean query into keywords
        query_words = self._clean_query(query)
        
        if not query_words:
            # If no valid query words, return first few documents
            top_docs = self.documents[:k]
        else:
            # Calculate relevance scores for all documents
            scored_docs = []
            for doc in self.documents:
                score = self._calculate_relevance_score(query_words, doc)
                scored_docs.append((score, doc))
            
            # Sort by score (highest first)
            scored_docs.sort(reverse=True, key=lambda x: x[0])
            
            # Get top k documents
            top_docs = [doc for score, doc in scored_docs[:k] if score > 0]
            
            # If no relevant docs found, return top k by default
            if not top_docs:
                top_docs = [doc for _, doc in scored_docs[:k]]
        
        print(f"🔍 Retrieved {len(top_docs)} relevant documents for query: '{query}'")

        # Build context
        context_parts = []
        for i, doc in enumerate(top_docs, 1):
            context_parts.append(f"Information {i}:\n{doc['content']}")

        return "\n\n".join(context_parts)

    def get_company_info(self) -> Dict[str, Any]:
        """Get basic company information."""
        data = self.load_company_data()
        company = data.get("company", {})
        return {
            "name": company.get("name", "Shyampari Edutech"),
            "location": company.get("location", "Pune, Maharashtra"),
        }


# ==================== HELPER FUNCTIONS ====================
def log_message(user_id, message, is_user=True, response=None, error=None):
    """Log message interactions to file"""
    timestamp = datetime.now().isoformat()
    log_entry = {
        'timestamp': timestamp,
        'user_id': user_id,
        'message_type': 'user' if is_user else 'ai',
        'message': message,
        'response': response,
        'error': error,
        'ip_address': request.remote_addr,
        'user_agent': request.headers.get('User-Agent', 'Unknown')
    }
    
    # Save to JSON log file
    log_file = logs_dir / f'chat_logs_{datetime.now().strftime("%Y-%m-%d")}.json'
    try:
        if log_file.exists():
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
            
    excep
