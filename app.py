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


# ==================== RAG SYSTEM CLASS ====================
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
            
    except Exception as e:
        logging.error(f"Failed to write to log file: {e}")
    
    # Also log to console
    if is_user:
        logging.info(f"User {user_id} ({request.remote_addr}): {message}")
    else:
        logging.info(f"AI Response to {user_id}: {response[:100] if response else 'None'}...")


def get_user_id():
    """Generate or retrieve user session ID"""
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id


# ==================== INITIALIZE SERVICES ====================
# Get Groq API key from environment variables
api_key = os.getenv('GROQ_API_KEY')

# Configure Groq
if api_key:
    client = Groq(api_key=api_key)
else:
    client = None

# Initialize RAG system
rag_system = None
if api_key:
    try:
        rag_system = RAGSystem(api_key)
        rag_system.build_vectorstore()
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        rag_system = None

# Debug information
print(f"🔍 Environment check:")
print(f"   GROQ_API_KEY from env: {'✅ Set' if api_key else '❌ Not found'}")
if api_key:
    print(f"   API Key prefix: {api_key[:7]}...")
    print(f"   RAG System: {'✅ Initialized' if rag_system else '❌ Failed to initialize'}")
else:
    print("   Please set GROQ_API_KEY environment variable")
print()


# ==================== API ROUTES ====================
@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'AI Assistant API',
        'status': 'running',
        'endpoints': {
            'chat': '/api/chat',
            'health': '/api/health',
            'rebuild': '/api/rebuild-vectorstore',
            'logs': '/api/logs',
            'contact': '/api/contact'
        }
    })


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        user_id = get_user_id()
        log_message(user_id, message, is_user=True)
        
        # Check if API key is set
        if not api_key:
            error_msg = 'Groq API key not configured. Please set GROQ_API_KEY environment variable.'
            log_message(user_id, message, is_user=False, error=error_msg)
            return jsonify({
                'error': error_msg,
                'success': False
            }), 500
        
        # Check if RAG system is initialized
        if not rag_system:
            error_msg = 'RAG system not initialized'
            log_message(user_id, message, is_user=False, error=error_msg)
            return jsonify({
                'error': error_msg,
                'success': False
            }), 500
        
        # Get company information
        company_info = rag_system.get_company_info()

        # Get company summary
        company_summary = rag_system.get_summary_document()

        # Search for relevant context
        try:
            relevant_context = rag_system.search_relevant_context(message, k=4)
            print(f"📚 Retrieved relevant context for query: {message}")
        except Exception as e:
            print(f"⚠️ RAG search failed: {e}")
            relevant_context = "Unable to retrieve relevant information from the knowledge base."
        
        # Create prompt for Groq
        prompt = f"""You are a helpful and professional AI assistant representing {company_info['name']}.
Your goal is to provide accurate and comprehensive answers about the company's services, offerings, and processes.

COMPANY OVERVIEW:
{company_summary}

USER QUESTION:
{message}

DETAILED INFORMATION FROM KNOWLEDGE BASE:
{relevant_context}

INSTRUCTIONS:
- Answer the user's question directly and accurately based on the information provided above.
- Be friendly, professional, and helpful in your responses.
- If the information is not available in the context, politely state that you don't have that specific information.
- Keep your responses clear, pointwise, and well-structured.
- Always maintain a helpful and supportive tone.

Please provide your response:"""

        # Generate response using Groq
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1024,
        )
        ai_response = chat_completion.choices[0].message.content
        
        # Log AI response
        log_message(user_id, message, is_user=False, response=ai_response)
        
        return jsonify({
            'response': ai_response,
            'success': True,
            'session_id': user_id
        })
        
    except Exception as e:
        error_msg = f'Failed to get AI response: {str(e)}'
        user_id = get_user_id()
        log_message(user_id, message if 'message' in locals() else 'Unknown', is_user=False, error=error_msg)
        print(f"Error: {str(e)}")
        return jsonify({
            'error': 'Failed to get AI response',
            'success': False
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    api_key_status = "configured" if api_key else "not configured"
    rag_status = "initialized" if rag_system else "not initialized"
    return jsonify({
        'status': 'healthy', 
        'message': 'AI Assistant API is running',
        'api_key': api_key_status,
        'rag_system': rag_status
    })


@app.route('/api/rebuild-vectorstore', methods=['POST'])
def rebuild_vectorstore():
    """Rebuild vector database"""
    try:
        if not api_key:
            return jsonify({
                'error': 'Groq API key not configured',
                'success': False
            }), 500
        
        global rag_system
        rag_system = RAGSystem(api_key)
        rag_system.build_vectorstore()
        
        return jsonify({
            'message': 'Vector database rebuilt successfully',
            'success': True
        })
        
    except Exception as e:
        print(f"Error rebuilding vectorstore: {str(e)}")
        return jsonify({
            'error': 'Failed to rebuild vector database',
            'success': False
        }), 500


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get chat logs with optional filtering"""
    try:
        date = request.args.get('date')
        user_id = request.args.get('user_id')
        limit = int(request.args.get('limit', 100))
        
        if date:
            log_file = logs_dir / f'chat_logs_{date}.json'
        else:
            log_file = logs_dir / f'chat_logs_{datetime.now().strftime("%Y-%m-%d")}.json'
        
        if not log_file.exists():
            return jsonify({
                'logs': [],
                'message': 'No logs found for the specified date',
                'success': True
            })
        
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        if user_id:
            logs = [log for log in logs if log.get('user_id') == user_id]
        
        logs = logs[-limit:]
        
        return jsonify({
            'logs': logs,
            'total_entries': len(logs),
            'date': date or datetime.now().strftime("%Y-%m-%d"),
            'success': True
        })
        
    except Exception as e:
        logging.error(f"Error retrieving logs: {e}")
        return jsonify({
            'error': 'Failed to retrieve logs',
            'success': False
        }), 500


@app.route('/api/contact', methods=['POST'])
def contact_submit():
    """Handle contact form submissions"""
    try:
        data = request.get_json(force=True)
        name = (data.get('name') or '').strip()
        company = (data.get('company') or '').strip()
        email = (data.get('email') or '').strip()
        message = (data.get('message') or '').strip()
        
        if not name or not email or not message:
            return jsonify({
                'success': False, 
                'error': 'name, email and message are required'
            }), 400
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'name': name,
            'company': company,
            'email': email,
            'message': message,
            'ip_address': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', 'Unknown')
        }
        
        contacts_file = logs_dir / f'contacts_{datetime.now().strftime("%Y-%m-%d")}.json'
        
        try:
            if contacts_file.exists():
                with open(contacts_file, 'r', encoding='utf-8') as f:
                    contacts = json.load(f)
            else:
                contacts = []
            
            contacts.append(entry)
            
            with open(contacts_file, 'w', encoding='utf-8') as f:
                json.dump(contacts, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"Failed to write contact file: {e}")
        
        return jsonify({'success': True, 'message': 'Contact received'})
        
    except Exception as e:
        logging.error(f"Error in contact_submit: {e}")
        return jsonify({
            'success': False, 
            'error': 'Failed to submit contact'
        }), 500


# ==================== VERCEL HANDLER ====================
# This is required for Vercel serverless deployment
app_handler = app


if __name__ == '__main__':
    print("🚀 Starting AI Assistant Backend with Groq + RAG...")
    print(f"📡 API Key Status: {'✅ Configured' if api_key else '❌ Not configured'}")
    print(f"🧠 RAG System: {'✅ Ready' if rag_system else '❌ Not ready'}")
    print("🌐 Server will be available at: http://localhost:5001")
    print("📋 API Endpoints:")
    print("   - POST /api/chat - Send message to AI (with RAG)")
    print("   - GET  /api/health - Health check")
    print("   - POST /api/rebuild-vectorstore - Rebuild vector database")
    print("   - GET  /api/logs - View chat logs")
    print("   - POST /api/contact - Receive contact form submissions")
    print("📝 Message logging is enabled - logs saved to logs/ directory")
    print("\n💡 Make sure to set GROQ_API_KEY environment variable")
    print("🔑 Get your free API key at: https://console.groq.com/keys")
    app.run(debug=True, host='0.0.0.0', port=5001)
