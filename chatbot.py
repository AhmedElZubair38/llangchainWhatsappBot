from flask import Flask, request, jsonify, session, render_template
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI
import pandas as pd
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'supersecretkey')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROGRAMS = {
    '1': 'Kids Program',
    '2': 'Adults Program',
    '3': 'Ladies-Only Aqua Fitness',
    '4': 'Baby & Toddler Program',
    '5': 'Special Needs Program'
}

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

def save_inquiry(data):
    df = pd.DataFrame([data])
    if not os.path.exists('inquiries.csv'):
        df.to_csv('inquiries.csv', index=False)
    else:
        df.to_csv('inquiries.csv', mode='a', header=False, index=False)

@app.route('/')
def chat_interface():
    # Initialize session
    session['session_id'] = str(uuid.uuid4())
    session['state'] = 'MAIN_MENU'
    return render_template('chat.html')

@app.route('/send_message', methods=['POST'])
def handle_message():
    user_input = request.json.get('message', '').strip()
    session_id = session.get('session_id')
    current_state = session.get('state', 'MAIN_MENU')
    
    response = process_message(user_input, current_state, session_id)
    session['state'] = response.get('new_state', 'MAIN_MENU')
    
    return jsonify(response)

def process_message(message, current_state, session_id):
    try:
        if message.lower() == 'menu':
            return {
                "text": get_main_menu()['text'],
                "options": get_main_menu()['options'],
                "new_state": 'MAIN_MENU'
            }
            
        if current_state == 'MAIN_MENU':
            return handle_main_menu(message)
        elif current_state == 'PROGRAM_INFO':
            return handle_program_info(message)
        elif current_state == 'BOOKING':
            return handle_booking(message, session_id)
        elif current_state == 'AI_QUERY':
            return handle_ai_query(message)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"text": "⚠️ An error occurred. Please try again or type 'menu'"}

def handle_main_menu(message):
    if message == '1':
        return {
            "text": "Choose program:",
            "options": [{"value": k, "label": v} for k, v in PROGRAMS.items()],
            "new_state": 'BOOKING_PROGRAM'
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
    elif message == '5':
        return {
            "text": "Ask me anything about our programs!",
            "new_state": 'AI_QUERY'
        }
    else:
        return get_main_menu()

def handle_program_info(message):
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
            "options": [{"value": "menu", "label": "Return to Menu"}]
        }
    else:
        return {
            "text": "Invalid choice. Please select 1-5",
            "options": [{"value": k, "label": v} for k, v in PROGRAMS.items()]
        }

def handle_booking(message, session_id):
    booking_data = session.get('booking_data', {})
    
    if session.get('booking_step') == 'PROGRAM_CHOICE':
        program = PROGRAMS.get(message)
        if program:
            booking_data['program'] = program
            session['booking_data'] = booking_data
            session['booking_step'] = 'GET_NAME'
            return {"text": "Please enter your full name:"}
        else:
            return {
                "text": "Invalid program selection. Please choose 1-5",
                "options": [{"value": k, "label": v} for k, v in PROGRAMS.items()]
            }
    
    elif session.get('booking_step') == 'GET_NAME':
        booking_data['name'] = message
        session['booking_data'] = booking_data
        session['booking_step'] = 'GET_PHONE'
        return {"text": "Please enter your phone number:"}
    
    elif session.get('booking_step') == 'GET_PHONE':
        booking_data['phone'] = message
        booking_data['timestamp'] = datetime.now()
        save_inquiry(booking_data)
        
        session.pop('booking_data', None)
        session.pop('booking_step', None)
        
        return {
            "text": ("📝 Booking Received!\n\n"
                    f"Program: {booking_data['program']}\n"
                    f"Name: {booking_data['name']}\n"
                    f"Phone: {booking_data['phone']}\n\n"
                    "We'll contact you shortly!"),
            "options": [{"value": "menu", "label": "Return to Menu"}]
        }

def handle_ai_query(message):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        ai_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "You're an assistant for Aquasprint Swimming Academy. Provide helpful responses about swimming programs, safety, and facilities. Keep answers concise."
            }, {
                "role": "user",
                "content": message
            }]
        )
        response_text = ai_response.choices[0].message.content
        return {
            "text": f"🤖 AI Agent:\n{response_text}",
            "options": [{"value": "menu", "label": "Return to Menu"}]
        }
    except Exception as e:
        logger.error(f"AI Query Failed: {e}")
        return {"text": "Our AI agent is currently busy. Please try again later."}

if __name__ == '__main__':
    app.run(port=5000)