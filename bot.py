from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI
import pandas as pd
import threading

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory session storage (replace with database in production)
user_sessions = {}

PROGRAMS = {
    '1': 'Kids Program',
    '2': 'Adults Program',
    '3': 'Ladies-Only Aqua Fitness',
    '4': 'Baby & Toddler Program',
    '5': 'Special Needs Program'
}

def get_main_menu():
    return (
        "👋 Welcome to Aquasprint Swimming Academy!\n\n"
        "Choose an option:\n"
        "1. Book a Class\n"
        "2. Program Information\n"
        "3. Location & Hours\n"
        "4. Contact Us\n"
        "5. Talk to AI Agent"
    )

def save_inquiry(data):
    lock = threading.Lock()
    with lock:
        df = pd.DataFrame([data])
        if not os.path.exists('inquiries.csv'):
            df.to_csv('inquiries.csv', index=False)
        else:
            df.to_csv('inquiries.csv', mode='a', header=False, index=False)


@app.route('/webhook', methods=['POST'])
def webhook():
    sender = request.values.get('From', '')
    message_body = request.values.get('Body', '').strip()
    resp = MessagingResponse()
    
    # Get or initialize session
    session = user_sessions.get(sender, {'state': 'MAIN_MENU'})
    
    try:
        if message_body.lower() == 'menu':
            session = {'state': 'MAIN_MENU'}
            resp.message(get_main_menu())
        elif session['state'] == 'MAIN_MENU':
            handle_main_menu(message_body, resp, session)
        elif session['state'] == 'PROGRAM_INFO':
            handle_program_info(message_body, resp, session)
        elif session['state'] == 'BOOKING':
            handle_booking(message_body, resp, session)
        elif session['state'] == 'AI_QUERY':
            handle_ai_query(message_body, resp, session)
            
        user_sessions[sender] = session
        
    except Exception as e:
        logger.error(f"Error: {e}")
        resp.message("⚠️ An error occurred. Please try again or type 'menu'")

    return str(resp)

def handle_main_menu(message, resp, session):
    if message == '1':
        session.update({'state': 'BOOKING', 'step': 'PROGRAM_CHOICE'})
        resp.message("Choose program:\n" + "\n".join([f"{k}. {v}" for k,v in PROGRAMS.items()]))
    elif message == '2':
        session.update({'state': 'PROGRAM_INFO'})
        resp.message("Choose program for details:\n" + "\n".join([f"{k}. {v}" for k,v in PROGRAMS.items()]))
    elif message == '3':
        resp.message(
            "🏊‍♂️ Aquasprint Swimming Academy\n\n"
            "📍 Location: The Sustainable City, Dubai\n"
            "⏰ Hours: Daily 6AM-10PM\n"
            "📞 +971542502761\n"
            "📧 info@aquasprint.ae\n\n"
            "Type 'menu' to return"
        )
    elif message == '5':
        session.update({'state': 'AI_QUERY'})
        resp.message("Ask me anything about our programs!")
    else:
        resp.message(get_main_menu())

def handle_program_info(message, resp, session):
    program = PROGRAMS.get(message)
    if program:
        # Add actual program details
        resp.message(f"{program} Details:\n4-14 years, 8 levels available\nType 'menu' to return")
    else:
        resp.message("Invalid choice. Please select 1-5")

def handle_booking(message, resp, session):
    if session.get('step') == 'PROGRAM_CHOICE':
        program = PROGRAMS.get(message)
        if program:
            session['program'] = program
            session['step'] = 'GET_NAME'
            resp.message("Please enter your full name:")
        else:
            resp.message("Invalid program selection. Please choose 1-5")
    
    elif session.get('step') == 'GET_NAME':
        session['name'] = message
        session['step'] = 'GET_PHONE'
        resp.message("Please enter your phone number:")
    
    elif session.get('step') == 'GET_PHONE':
        session['phone'] = message
        save_inquiry({
            'timestamp': datetime.now(),
            'name': session['name'],
            'phone': session['phone'],
            'program': session['program']
        })
        resp.message(
            f"📝 Booking Received!\n\n"
            f"Program: {session['program']}\n"
            f"Name: {session['name']}\n"
            f"Phone: {session['phone']}\n\n"
            "We'll contact you shortly!\n"
            "Type 'menu' to return"
        )
        # user_sessions.pop(sender, None)

def handle_ai_query(message, resp, session):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    ai_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}]
    )
    response_text = ai_response.choices[0].message.content
    resp.message(response_text + "\n\nType 'menu' to return")

if __name__ == '__main__':
    app.run(port=5000)