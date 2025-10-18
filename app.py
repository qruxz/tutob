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
import psycopg2
from psycopg2.extras import RealDictCursor

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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


# ==================== NEON DB HELPER ====================
class NeonDB:
    """Helper class for Neon PostgreSQL database operations."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.connection_string)
    
    def init_tables(self):
        """Initialize database tables."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Create documents table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata JSONB,
                        keywords TEXT[],
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create chat logs table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_logs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        user_id VARCHAR(255),
                        message_type VARCHAR(50),
                        message TEXT,
                        response TEXT,
                        error TEXT,
                        ip_address VARCHAR(100),
                        user_agent TEXT
                    )
                """)
                
                # Create contacts table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS contacts (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        name VARCHAR(255),
                        company VARCHAR(255),
                        email VARCHAR(255),
                        message TEXT,
                        ip_address VARCHAR(100),
                        user_agent TEXT
                    )
                """)
                
                conn.commit()
                print("✅ Database tables initialized successfully!")
        except Exception as e:
            conn.rollback()
            print(f"❌ Error initializing tables: {e}")
            raise
        finally:
            conn.close()
    
    def store_documents(self, documents: List[Dict[str, Any]]):
        """Store documents in database."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Clear existing documents
                cur.execute("DELETE FROM documents")
                
                # Insert new documents
                for doc in documents:
                    cur.execute("""
                        INSERT INTO documents (content, metadata, keywords)
                        VALUES (%s, %s, %s)
                    """, (
                        doc['content'],
                        json.dumps(doc.get('metadata', {})),
                        doc.get('keywords', [])
                    ))
                
                conn.commit()
                print(f"✅ Stored {len(documents)} documents in Neon DB!")
        except Exception as e:
            conn.rollback()
            print(f"❌ Error storing documents: {e}")
            raise
        finally:
            conn.close()
    
    def search_documents(self, query_words: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Search documents using keyword matching."""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if not query_words:
                    # Return random documents if no query
                    cur.execute("SELECT * FROM documents ORDER BY RANDOM() LIMIT %s", (limit,))
                else:
                    # Search using keyword matching
                    # Calculate score based on keyword matches
                    query = """
                        SELECT *, 
                        (
                            SELECT COUNT(*) 
                            FROM unnest(keywords) AS kw 
                            WHERE kw = ANY(%s)
                        ) AS keyword_score,
                        (
                            SELECT COUNT(*) 
                            FROM unnest(%s) AS qw 
                            WHERE position(lower(qw) IN lower(content)) > 0
                        ) AS content_score
                        FROM documents
                        ORDER BY keyword_score DESC, content_score DESC
                        LIMIT %s
                    """
                    cur.execute(query, (query_words, query_words, limit))
                
                results = cur.fetchall()
                return [dict(row) for row in results]
        finally:
            conn.close()
    
    def log_chat(self, user_id: str, message: str, is_user: bool = True, response: str = None, error: str = None):
        """Log chat message to database."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_logs (user_id, message_type, message, response, error, ip_address, user_agent)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id,
                    'user' if is_user else 'ai',
                    message,
                    response,
                    error,
                    request.remote_addr,
                    request.headers.get('User-Agent', 'Unknown')
                ))
                conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error(f"Failed to log chat: {e}")
        finally:
            conn.close()
    
    def save_contact(self, name: str, email: str, message: str, company: str = None):
        """Save contact form submission."""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO contacts (name, company, email, message, ip_address, user_agent)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    name,
                    company,
                    email,
                    message,
                    request.remote_addr,
                    request.headers.get('User-Agent', 'Unknown')
                ))
                conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error(f"Failed to save contact: {e}")
            raise
        finally:
            conn.close()
    
    def get_logs(self, date: str = None, user_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get chat logs from database."""
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = "SELECT * FROM chat_logs WHERE 1=1"
                params = []
                
                if date:
                    query += " AND DATE(timestamp) = %s"
                    params.append(date)
                
                if user_id:
                    query += " AND user_id = %s"
                    params.append(user_id)
                
                query += " ORDER BY timestamp DESC LIMIT %s"
                params.append(limit)
                
                cur.execute(query, params)
                results = cur.fetchall()
                return [dict(row) for row in results]
        finally:
            conn.close()


# ==================== RAG SYSTEM CLASS ====================
class RAGSystem:
    """RAG System using Neon PostgreSQL for storage."""

    def __init__(self, api_key: str, neon_db: NeonDB):
        self.api_key = api_key
        self.neon_db = neon_db
        self.company_summary = ""

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
                "keywords": ["parent", "instruction", "guide", "how", "steps"]
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
        """Builds the document store in Neon DB."""
        print("🔧 Building document store in Neon DB...")

        data = self.load_company_data()

        # Generate and store the summary
        self.company_summary = self._generate_summary_text(data)
        print("✅ Company summary generated and cached.")

        # Create documents
        documents = self._create_documents_from_data(data)
        
        # Store in Neon DB
        self.neon_db.store_documents(documents)
        
        print(f"📄 Created and stored {len(documents)} documents in Neon DB.")
        print("✅ Document store built successfully!")

    def get_summary_document(self) -> str:
        """Returns the cached high-level summary text."""
        return self.company_summary

    def _clean_query(self, query: str) -> List[str]:
        """Clean and tokenize query into keywords."""
        query = re.sub(r'[^\w\s]', ' ', query.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'what', 'how', 'can', 'do', 'does'}
        words = [w for w in query.split() if w and w not in stop_words]
        return words

    def search_relevant_context(self, query: str, k: int = 5) -> str:
        """Search for relevant context using Neon DB."""
        # Clean query into keywords
        query_words = self._clean_query(query)
        
        # Search in Neon DB
        results = self.neon_db.search_documents(query_words, limit=k)
        
        print(f"🔍 Retrieved {len(results)} relevant documents from Neon DB for query: '{query}'")

        # Build context
        context_parts = []
        for i, doc in enumerate(results, 1):
            context_parts.append(f"Information {i}:\n{doc['content']}")

        return "\n\n".join(context_parts) if context_parts else "No relevant information found."

    def get_company_info(self) -> Dict[str, Any]:
        """Get basic company information."""
        data = self.load_company_data()
        company = data.get("company", {})
        return {
            "name": company.get("name", "Shyampari Edutech"),
            "location": company.get("location", "Pune, Maharashtra"),
        }


# ==================== HELPER FUNCTIONS ====================
def get_user_id():
    """Generate or retrieve user session ID"""
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id


# ==================== INITIALIZE SERVICES ====================
# Get environment variables
api_key = os.getenv('GROQ_API_KEY')
neon_connection_string = os.getenv('NEON_DATABASE_URL')

# Configure Groq
if api_key:
    client = Groq(api_key=api_key)
else:
    client = None

# Initialize Neon DB
neon_db = None
if neon_connection_string:
    try:
        neon_db = NeonDB(neon_connection_string)
        neon_db.init_tables()
        print("✅ Neon DB initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize Neon DB: {e}")
        neon_db = None

# Initialize RAG system
rag_system = None
if api_key and neon_db:
    try:
        rag_system = RAGSystem(api_key, neon_db)
        rag_system.build_vectorstore()
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        rag_system = None

# Debug information
print(f"🔍 Environment check:")
print(f"   GROQ_API_KEY: {'✅ Set' if api_key else '❌ Not found'}")
print(f"   NEON_DATABASE_URL: {'✅ Set' if neon_connection_string else '❌ Not found'}")
print(f"   Neon DB: {'✅ Connected' if neon_db else '❌ Not connected'}")
print(f"   RAG System: {'✅ Initialized' if rag_system else '❌ Failed to initialize'}")
print()


# ==================== API ROUTES ====================
@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'AI Assistant API - Groq + Neon DB',
        'status': 'running',
        'technology': 'Groq LLM + Neon PostgreSQL',
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
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        user_id = get_user_id()
        
        # Log to Neon DB
        if neon_db:
            neon_db.log_chat(user_id, message, is_user=True)
        
        # Check if API key is set
        if not api_key:
            error_msg = 'Groq API key not configured.'
            if neon_db:
                neon_db.log_chat(user_id, message, is_user=False, error=error_msg)
            return jsonify({'error': error_msg, 'success': False}), 500
        
        # Check if RAG system is initialized
        if not rag_system:
            error_msg = 'RAG system not initialized'
            if neon_db:
                neon_db.log_chat(user_id, message, is_user=False, error=error_msg)
            return jsonify({'error': error_msg, 'success': False}), 500
        
        # Get company information
        company_info = rag_system.get_company_info()
        company_summary = rag_system.get_summary_document()

        # Search for relevant context
        try:
            relevant_context = rag_system.search_relevant_context(message, k=4)
        except Exception as e:
            print(f"⚠️ RAG search failed: {e}")
            relevant_context = "Unable to retrieve relevant information."
        
        # Create prompt for Groq
        prompt = f"""You are a helpful and professional AI assistant representing {company_info['name']}.

COMPANY OVERVIEW:
{company_summary}

USER QUESTION:
{message}

DETAILED INFORMATION:
{relevant_context}

INSTRUCTIONS:
- Answer directly and accurately based on the information provided.
- Be friendly, professional, and helpful.
- If information is not available, politely state that.
- Keep responses clear, pointwise, and well-structured.

Please provide your response:"""

        # Generate response using Groq
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1024,
        )
        ai_response = chat_completion.choices[0].message.content
        
        # Log AI response to Neon DB
        if neon_db:
            neon_db.log_chat(user_id, message, is_user=False, response=ai_response)
        
        return jsonify({
            'response': ai_response,
            'success': True,
            'session_id': user_id
        })
        
    except Exception as e:
        error_msg = f'Failed to get AI response: {str(e)}'
        logging.error(error_msg)
        return jsonify({'error': 'Failed to get AI response', 'success': False}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'message': 'AI Assistant API is running',
        'technology': 'Groq + Neon PostgreSQL',
        'api_key': 'configured' if api_key else 'not configured',
        'database': 'connected' if neon_db else 'not connected',
        'rag_system': 'initialized' if rag_system else 'not initialized'
    })


@app.route('/api/rebuild-vectorstore', methods=['POST'])
def rebuild_vectorstore():
    """Rebuild document store"""
    try:
        if not api_key or not neon_db:
            return jsonify({'error': 'Services not configured', 'success': False}), 500
        
        global rag_system
        rag_system = RAGSystem(api_key, neon_db)
        rag_system.build_vectorstore()
        
        return jsonify({'message': 'Document store rebuilt successfully', 'success': True})
        
    except Exception as e:
        logging.error(f"Error rebuilding: {str(e)}")
        return jsonify({'error': 'Failed to rebuild', 'success': False}), 500


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get chat logs from Neon DB"""
    try:
        if not neon_db:
            return jsonify({'error': 'Database not connected', 'success': False}), 500
        
        date = request.args.get('date')
        user_id = request.args.get('user_id')
        limit = int(request.args.get('limit', 100))
        
        logs = neon_db.get_logs(date=date, user_id=user_id, limit=limit)
        
        return jsonify({
            'logs': logs,
            'total_entries': len(logs),
            'success': True
        })
        
    except Exception as e:
        logging.error(f"Error retrieving logs: {e}")
        return jsonify({'error': 'Failed to retrieve logs', 'success': False}), 500


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
            return jsonify({'success': False, 'error': 'name, email and message are required'}), 400
        
        if neon_db:
            neon_db.save_contact(name, email, message, company)
        
        return jsonify({'success': True, 'message': 'Contact received'})
        
    except Exception as e:
        logging.error(f"Error in contact_submit: {e}")
        return jsonify({'success': False, 'error': 'Failed to submit contact'}), 500


# Vercel handler
app_handler = app


if __name__ == '__main__':
    print("🚀 Starting AI Assistant Backend - Groq + Neon DB...")
    print(f"📡 API Key: {'✅ Configured' if api_key else '❌ Not configured'}")
    print(f"🗄️  Neon DB: {'✅ Connected' if neon_db else '❌ Not connected'}")
    print(f"🧠 RAG System: {'✅ Ready' if rag_system else '❌ Not ready'}")
    print("🌐 Server: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
