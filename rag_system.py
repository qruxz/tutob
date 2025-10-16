import os
import json
from pathlib import Path
from typing import List, Dict, Any


class RAGSystem:
    """
    Simple RAG System using only Groq - no external dependencies for embeddings.
    Uses basic text matching for document retrieval.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.documents = []
        self.company_summary = ""

    def load_company_data(self) -> Dict[str, Any]:
        """Load company data from data.json file located next to this file."""
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
        """
        Creates a list of documents from the company data.
        Each major section becomes its own document.

        Returns list of dicts with 'content' and 'metadata' keys.
        """
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
            documents.append(
                {"content": company_content, "metadata": {"type": "company_info"}}
            )

        # Each Service as separate document
        for service in data.get("services", []):
            service_content = f"""Service: {service.get('name', 'N/A')}
Description: {service.get('description', 'N/A')}"""
            documents.append(
                {
                    "content": service_content,
                    "metadata": {"type": "service", "service_name": service.get("name", "N/A")},
                }
            )

        # Each "Why Choose Us" point as separate document
        for reason in data.get("why_choose_us", []):
            reason_content = f"""Reason: {reason.get('title', 'N/A')}
Details: {reason.get('description', 'N/A')}"""
            documents.append(
                {
                    "content": reason_content,
                    "metadata": {"type": "why_choose_us", "title": reason.get("title", "N/A")},
                }
            )

        # Demo Process as single document
        demo = data.get("demo_process", {})
        if demo:
            steps = demo.get("steps", [])
            demo_content = f"""Demo Process:
Steps: {'; '.join(steps)}
Demo Fee: {demo.get('demo_fee', 'N/A')}
Profile Finalization Time: {demo.get('profile_finalization_time', 'N/A')}
Demo Duration: {demo.get('demo_duration', 'N/A')}"""
            documents.append({"content": demo_content, "metadata": {"type": "demo_process"}})

        # Parent Instructions as single document
        parent_instructions = data.get("parent_instructions", [])
        if parent_instructions:
            documents.append(
                {
                    "content": f"Parent Instructions:\n"
                    + "\n".join([f"- {inst}" for inst in parent_instructions]),
                    "metadata": {"type": "parent_instructions"},
                }
            )

        # Tutor Instructions as single document
        tutor_instructions = data.get("tutor_instructions", [])
        if tutor_instructions:
            documents.append(
                {
                    "content": f"Tutor Instructions:\n"
                    + "\n".join([f"- {inst}" for inst in tutor_instructions]),
                    "metadata": {"type": "tutor_instructions"},
                }
            )

        # Student Flow as single document
        student_flow = data.get("student_flow", [])
        if student_flow:
            documents.append(
                {
                    "content": f"Student Flow:\n"
                    + "\n".join([f"{i+1}. {step}" for i, step in enumerate(student_flow)]),
                    "metadata": {"type": "student_flow"},
                }
            )

        # Tutor Flow as single document
        tutor_flow = data.get("tutor_flow", [])
        if tutor_flow:
            documents.append(
                {
                    "content": f"Tutor Flow:\n"
                    + "\n".join([f"{i+1}. {step}" for i, step in enumerate(tutor_flow)]),
                    "metadata": {"type": "tutor_flow"},
                }
            )

        # Matching System as single document
        matching = data.get("matching_system", {})
        if matching:
            factors = matching.get("factors", [])
            matching_content = f"""Matching System:
Type: {matching.get('type', 'N/A')}
Factors Considered: {', '.join(factors)}
Description: {matching.get('description', 'N/A')}"""
            documents.append({"content": matching_content, "metadata": {"type": "matching_system"}})

        # Payment Info as single document
        payment = data.get("payment_info", {})
        if payment:
            payment_content = f"""Payment Information:
Payment Options: {payment.get('payment_options', 'N/A')}
Package Options: {payment.get('package_options', 'N/A')}
Demo Fee: {payment.get('demo_fee', 'N/A')}
Policy: {payment.get('policy', 'N/A')}"""
            documents.append({"content": payment_content, "metadata": {"type": "payment_info"}})

        # Boards and Class Modes as single document
        boards = data.get("boards_supported", [])
        modes = data.get("class_modes", [])
        if boards or modes:
            documents.append(
                {
                    "content": f"Educational Boards Supported: {', '.join(boards)}\nClass Modes Available: {', '.join(modes)}",
                    "metadata": {"type": "boards_and_modes"},
                }
            )

        # Each FAQ as separate document
        for faq in data.get("faq", []):
            faq_content = f"""Question: {faq.get('question', 'N/A')}
Answer: {faq.get('answer', 'N/A')}"""
            documents.append(
                {
                    "content": faq_content,
                    "metadata": {"type": "faq", "question": faq.get("question", "N/A")},
                }
            )

        return documents

    def build_vectorstore(self):
        """
        Builds the document store by loading and processing data.
        """
        print("🔧 Building document store...")

        data = self.load_company_data()

        # Generate and store the summary
        self.company_summary = self._generate_summary_text(data)
        print("✅ Company summary generated and cached.")

        # Create documents
        self.documents = self._create_documents_from_data(data)
        print(f"📄 Created {len(self.documents)} documents.")
        print("✅ Document store built successfully!")

    def get_summary_document(self) -> str:
        """Returns the cached high-level summary text."""
        return self.company_summary

    def _calculate_relevance_score(self, query: str, document: str) -> float:
        """
        Simple keyword-based relevance scoring.
        Returns a score based on how many query words appear in the document.
        """
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        if not query_words:
            return 0.0
        
        # Count matching words
        matches = len(query_words.intersection(doc_words))
        
        # Normalize by query length
        score = matches / len(query_words)
        
        return score

    def search_relevant_context(self, query: str, k: int = 5) -> str:
        """
        Search for relevant context from the documents using simple text matching.

        Args:
            query: User query
            k: Number of relevant documents to return

        Returns:
            Relevant context string (concatenated top-k documents)
        """
        if not self.documents:
            raise ValueError("Documents not loaded. Call build_vectorstore() first.")

        # Calculate relevance scores for all documents
        scored_docs = []
        for doc in self.documents:
            score = self._calculate_relevance_score(query, doc["content"])
            scored_docs.append((score, doc))
        
        # Sort by score (highest first)
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        # Get top k documents
        top_docs = [doc for score, doc in scored_docs[:k] if score > 0]
        
        # If no relevant docs found, return all documents (fallback)
        if not top_docs:
            top_docs = [doc for _, doc in scored_docs[:k]]
        
        print(f"🔍 Retrieved {len(top_docs)} relevant documents for the query.")

        # Build context parts
        context_parts = []
        for i, doc in enumerate(top_docs, 1):
            context_parts.append(f"Relevant Information {i}:\n{doc['content']}")

        return "\n\n".join(context_parts)

    def get_company_info(self) -> Dict[str, Any]:
        """Get basic company information for prompts"""
        data = self.load_company_data()
        company = data.get("company", {})
        return {
            "name": company.get("name", "Shyampari Edutech"),
            "location": company.get("location", "Pune, Maharashtra"),
        }