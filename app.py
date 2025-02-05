from flask import Flask, request, jsonify, session, render_template
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import uuid
import threading
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'supersecretkey')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

csv_lock = threading.Lock()

PROGRAMS = {
    '1': 'Kids Program',
    '2': 'Adults Program',
    '3': 'Ladies-Only Aqua Fitness',
    '4': 'Baby & Toddler Program',
    '5': 'Special Needs Program'
}

uid = 'postgres'
pwd = 'ahmed'
server = "172.27.249.6"
database = "sample"

def get_main_menu():
    return {
        "text": "👋 Welcome to Aquasprint Swimming Academy!\n\nChoose an option:",
        "options": [
            {"value": "1", "label": "Book a Class"},
            {"value": "2", "label": "Program Information"},
            {"value": "3", "label": "Location & Hours"},
            {"value": "4", "label": "Contact Us"},
            {"value": "5", "label": "Talk to AI Agent"}
        ]
    }

# def save_inquiry(data):
#     """ Save inquiry data to CSV file in a thread-safe manner """
#     with csv_lock:
#         try:
#             df = pd.DataFrame([data])
#             file_exists = os.path.exists('data/inquiries.csv')
#             df.to_csv('data/inquiries.csv', mode='a', header=not file_exists, index=False)
#         except Exception as e:
#             logger.error(f"Save failed: {str(e)}")
#             raise

import psycopg2
from psycopg2 import sql

def save_inquiry(data):
    """ Save inquiry data to PostgreSQL database """
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=database,
            user=uid,
            password=pwd,
            host=server
        )
        cur = conn.cursor()
        
        query = sql.SQL("""
            INSERT INTO inquiries (program, name, phone, email, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        """)
        
        cur.execute(query, (
            data['program'],
            data['name'],
            data['phone'],
            data['email'],
            data['timestamp']
        ))
        
        conn.commit()
        cur.close()
    except Exception as e:
        logger.error(f"Save failed: {str(e)}")
        raise
    finally:
        if conn is not None:
            conn.close()

@app.route('/')
def chat_interface():
    session['session_id'] = str(uuid.uuid4())
    session['state'] = 'MAIN_MENU'
    return render_template('chat.html')

@app.route('/send_message', methods=['POST'])
def handle_message():
    user_input = request.json.get('message', '').strip()
    session_id = session.get('session_id')
    current_state = session.get('state', 'MAIN_MENU')

    response = process_message(user_input, current_state, session_id)

    if response is None:
        return jsonify({
            "text": "⚠️ An error occurred while processing your request. Please try again or type 'menu'"
        })

    if "new_state" in response:
        session['state'] = response.get('new_state', 'MAIN_MENU')

    return jsonify(response)

def process_message(message, current_state, session_id):
    try:
        logger.info(f"Processing message: {message}, Current state: {current_state}, Session ID: {session_id}")

        if message.lower() == 'menu':
            return {
                "text": get_main_menu()['text'],
                "options": get_main_menu()['options'],
                "new_state": 'MAIN_MENU'
            }
            
        if current_state == 'MAIN_MENU':
            return handle_main_menu(message)
        elif current_state == 'PROGRAM_SELECTION':
            return handle_program_selection(message)
        elif current_state == 'PROGRAM_INFO':
            return handle_program_info(message)
        elif current_state == 'BOOKING_PROGRAM':
            return handle_booking(message)
        elif current_state == 'AI_QUERY':
            return handle_ai_query(message)
        else:
            logger.error(f"Unknown state: {current_state}")
            return {"text": "⚠️ Unknown state. Please try again or type 'menu'"}
            
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return {"text": "⚠️ An error occurred. Please try again or type 'menu'"}

def handle_main_menu(message):
    if message == '1':
        return {
            "text": "Choose program:",
            "options": [{"value": k, "label": v} for k, v in PROGRAMS.items()],
            "new_state": 'PROGRAM_SELECTION'
        }
    elif message == '2':
        return {
            "text": "Choose program for details:",
            "options": [{"value": k, "label": v} for k, v in PROGRAMS.items()],
            "new_state": 'PROGRAM_INFO'
        }
    elif message == '3':
        return {
            "text": ("🏊‍♂️ Aquasprint Swimming Academy\n\n"
                     "📍 Location: The Sustainable City, Dubai\n"
                     "⏰ Hours: Daily 6AM-10PM\n"
                     "📞 +971542502761\n"
                     "📧 info@aquasprint.ae"),
            "options": [{"value": "menu", "label": "Return to Menu"}]
        }
    elif message == '4':
        return {
            "text": ("📞 Contact Us:\n"
                     "Call us at +971542502761\n"
                     "Email: info@aquasprint.ae"),
            "options": [{"value": "menu", "label": "Return to Menu"}]
        }
    elif message == '5':
        return {
            "text": "Ask me anything about our programs!",
            "new_state": 'AI_QUERY'
        }
    else:
        return get_main_menu()

def handle_program_info(message):
    """ Handles the display of program information based on user choice """
    program = PROGRAMS.get(message)
    if program:
        details = {
            '1': "👶 Kids Program (4-14 years)\n- 8 skill levels\n- Certified instructors",
            '2': "🏊 Adults Program\n- Beginner to advanced\n- Flexible scheduling",
            '3': "🚺 Ladies-Only Aqua Fitness\n- Women-only sessions\n- Full-body workout",
            '4': "👶👨👩 Baby & Toddler\n- Parent-child classes\n- Water safety basics",
            '5': "🌟 Special Needs Program\n- Adapted curriculum\n- Individual attention"
        }.get(message, "Program details not available")
        
        return {
            "text": f"{program} Details:\n{details}",
            "options": [{"value": "menu", "label": "Return to Menu"}],
            "new_state": "MAIN_MENU"
        }
    else:
        return {
            "text": "Invalid choice. Please select 1-5",
            "options": [{"value": k, "label": v} for k, v in PROGRAMS.items()],
            "new_state": 'PROGRAM_INFO'
        }

def handle_program_selection(message):
    """ Prepares for booking by capturing the selected program """
    program = PROGRAMS.get(message)
    if program:
        session['booking_data'] = {'program': program}
        session['booking_step'] = 'GET_NAME'
        return {
            "text": f"Selected program: {program}\nWhat's your full name?",
            "new_state": 'BOOKING_PROGRAM'
        }
    else:
        return {
            "text": "Invalid program selection. Please choose 1-5",
            "options": [{"value": k, "label": v} for k, v in PROGRAMS.items()],
            "new_state": 'PROGRAM_SELECTION'
        }

def handle_booking(message):
    """ Handles the multi-step booking process """
    try:
        current_step = session.get('booking_step', 'GET_NAME')
        booking_data = session.get('booking_data', {})

        logger.info(f"Booking Step: {current_step}, Message: {message}")

        if current_step == 'GET_NAME':
            booking_data['name'] = message
            session['booking_step'] = 'GET_PHONE'
            session['booking_data'] = booking_data
            return {"text": "📱 What's your phone number?"}

        elif current_step == 'GET_PHONE':
            booking_data['phone'] = message
            session['booking_step'] = 'GET_EMAIL'
            session['booking_data'] = booking_data
            return {"text": "📧 What's your email address?"}

        elif current_step == 'GET_EMAIL':
            booking_data['email'] = message
            booking_data['timestamp'] = datetime.now().isoformat()

            save_inquiry(booking_data)

            confirmation = (
                "✅ Booking confirmed!\n"
                f"Program: {booking_data.get('program')}\n"
                f"Name: {booking_data.get('name')}\n"
                f"Phone: {booking_data.get('phone')}\n"
                f"Email: {booking_data.get('email')}\n\n"
                "We'll contact you soon!"
            )

            session.pop('booking_data', None)
            session.pop('booking_step', None)

            return {
                "text": confirmation,
                "options": [{"value": "menu", "label": "Return to Menu"}],
                "new_state": 'MAIN_MENU'
            }

        return {"text": "Invalid booking step. Type 'menu' to start over."}

    except Exception as e:
        logger.error(f"Booking error: {str(e)}")
        return {"text": "⚠️ Booking failed. Type 'menu' to restart."}

def handle_ai_query(message):
    """Uses RAG with Ollama API to handle AI queries about swimming programs"""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="chroma_db"
        )
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})

        docs = retriever.invoke(message)
        
        knowledge = "\n\n".join([doc.page_content.strip() for doc in docs])
        knowledge += "\n\nEnd of knowledge base."

        llm = ChatOpenAI(
            model="deepseek-llm",
            base_url="http://172.27.240.1:11434/v1",
            verbose=True,
            temperature=0.1
        )

        messages = [
            SystemMessage(
            content=f"""You're an assistant for Aquasprint Swimming Academy. Follow these rules:
            1. Answer ONLY using the knowledge base below
            2. Be concise and professional
            3. If unsure, say "I don't have that information"
            4. Never make up answers

            Knowledge Base:
            {knowledge}"""
            ),
            HumanMessage(content=message)
        ]

        ai_response = llm.invoke(messages)
        response_text = ai_response.content

        return {
            "text": f"🤖 AI Agent:\n{response_text}",
            "options": [{"value": "menu", "label": "Return to Menu"}]
        }
    
    except Exception as e:
        logger.error(f"AI Query Failed: {e}")
        return {"text": "Our AI agent is currently busy. Please try again later."}


if __name__ == '__main__':
    app.run(port=5000)