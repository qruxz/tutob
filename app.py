from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from rag_system import RAGSystem
import logging
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
frontend_url = "https://tutof.vercel.app"

CORS(app, resources={r"/*": {"origins": frontend_url}})

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
    rag_system = RAGSystem(api_key)
    rag_system.build_vectorstore()

# Debug information
print(f"🔍 Environment check:")
print(f"   GROQ_API_KEY from env: {'✅ Set' if api_key else '❌ Not found'}")
if api_key:
    print(f"   API Key prefix: {api_key[:7]}...")
    print(f"   RAG System: {'✅ Initialized' if rag_system else '❌ Failed to initialize'}")
else:
    print("   Please set GROQ_API_KEY environment variable")
print()

@app.route('/api/chat', methods=['POST'])
def chat():
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
