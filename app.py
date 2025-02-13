import pandas as pd
import uuid
import threading
import re
import os
import requests
import logging
import psycopg2
from langchain_core.tools import tool
from fuzzywuzzy import process
from psycopg2 import sql
from typing import Optional, Dict, List
from dotenv import load_dotenv
from datetime import datetime
from pydantic import BaseModel, Field
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.output_parsers import PydanticOutputParser
from flask import Flask, request, jsonify, session, render_template

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'supersecretkey')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

csv_lock = threading.Lock()

PROGRAMS = {
    'Kids Program': 'Kids Program',
    'Adults Program': 'Adults Program',
    'Ladies-Only Aqua Fitness': 'Ladies-Only Aqua Fitness',
    'Baby & Toddler Program': 'Baby & Toddler Program',
    'Special Needs Program': 'Special Needs Program'
}

uid = 'postgres'
pwd = 'ahmed'
server = "172.27.249.6"
database = "sample"

def get_main_menu():
    return {
        "text": "👋 Welcome to Aquasprint Swimming Academy!\n\nChoose an option:",
        "options": [
            {"value": "Book a Class", "label": "Book a Class"},
            {"value": "Program Information", "label": "Program Information"},
            {"value": "Location & Hours", "label": "Location & Hours"},
            {"value": "Contact Us", "label": "Contact Us"},
            {"value": "Talk to AI Agent", "label": "Talk to AI Agent"}
        ]
    }

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


data_store = []

@app.route('/add_inquiry', methods=['POST'])
def add_inquiry():
    """ Exposes data as an API endpoint """
    data = request.json  # Assuming the data is sent as JSON
    
    # Store the inquiry in memory
    data_store.append(data)
    return jsonify({"message": "Data saved successfully!"}), 200


@app.route('/get_inquiries', methods=['GET'])
def get_inquiries():
    """ Fetch all stored inquiries via API """
    return jsonify(data_store), 200


class BookingInfo(BaseModel):
    program: Optional[str] = Field(None, description="The swimming program name")
    name: Optional[str] = Field(None, description="Customer's full name")
    phone: Optional[str] = Field(None, description="Customer's phone number")
    email: Optional[str] = Field(None, description="Customer's email address")

@tool
def extract_booking_info(query: str) -> BookingInfo:
    """Extract booking information from a natural language query using LangChain. Extracts only explicitly stated booking details. Prevents hallucination of missing data. IF INFORMATION NOT PROVIDED YOU STRICTLY RETURN `NONE` OKAY"""

    llm = ChatOpenAI(
        model="deepseek-llm",
        base_url="http://172.27.240.1:11434/v1",
        temperature=0.1
    )
    
    parser = PydanticOutputParser(pydantic_object=BookingInfo)
    
    prompt = f"""
    Extract booking information from the following query. If information is not present, return `None` for that field.
    
    Query: {query}
    
    Extract these fields:
    - `program`: The program name from this list: ["Kids Program", "Adults Program", "Ladies-Only Aqua Fitness", "Baby & Toddler Program", "Special Needs Program"].
      - If the user provides a similar name or has a typo, infer the correct program.
      - If no program is mentioned, return `None`.
    - `name`: Full name of the user. If not provided, return `None`.
    - `phone`: Extract a valid phone number. If not found, return `None`.
    - `email`: Extract a valid email address. If not found, return `None`.

    - ONLY return details that are explicitly mentioned.
    - If a detail is missing, set it to `None`.

    **Return ONLY a JSON object, formatted exactly as follows (no extra text):**

    ```json
    {{
        "program": "Program Name if given or None",
        "name": "Full Name if given or None",
        "phone": "Phone Number if given or None",
        "email": "Email Address if given or None"
    }}
    ```

    {parser.get_format_instructions()}
    """

    messages = [
        SystemMessage(content="You are a structured data extraction assistant. You return only JSON responses."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)

    print(response.content)

    try:
        return parser.parse(response.content.strip())  # Strip any leading/trailing whitespace
    except Exception as e:
        logger.error(f"Parsing error in extract_booking_info: {e}")
        return BookingInfo(program=None, name=None, phone=None, email=None)  # Return a blank BookingInfo object on failure


def get_missing_info(booking_info: BookingInfo) -> list:
    """Identify missing required booking information"""

    missing = []

    if not booking_info.program:
        missing.append('program')
    if not booking_info.name:
        missing.append('name')
    if not booking_info.phone:
        missing.append('phone')
    if not booking_info.email:
        missing.append('email')

    return missing

@app.route('/')
def chat_interface():
    session['session_id'] = str(uuid.uuid4())

    session['id'] = 1
    session['name'] = "Ahmed ElZubair"
    
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
    if message == 'Book a Class':
        return {
            "text": "Choose program:",
            "options": [
                {"value": v, "label": v} for k, v in PROGRAMS.items()
            ],
            "new_state": 'PROGRAM_SELECTION'
        }
    elif message == 'Program Information':
        return {
            "text": "Choose program for details:",
            "options": [
                {"value": v, "label": v} for k, v in PROGRAMS.items()
            ],
            "new_state": 'PROGRAM_INFO'
        }
    elif message == 'Location & Hours':
        return {
            "text": ("🏊‍♂️ Aquasprint Swimming Academy\n\n"
                     "📍 Location: The Sustainable City, Dubai\n"
                     "⏰ Hours: Daily 6AM-10PM\n"
                     "📞 +971542502761\n"
                     "📧 info@aquasprint.ae"),
            "options": [{"value": "menu", "label": "Return to Menu"}]
        }
    elif message == 'Contact Us':
        return {
            "text": ("📞 Contact Us:\n"
                     "Call us at +971542502761\n"
                     "Email: info@aquasprint.ae"),
            "options": [{"value": "menu", "label": "Return to Menu"}]
        }
    elif message == 'Talk to AI Agent':
        return {
            "text": "Hi! Ask me anything about our programs!",
            "new_state": 'AI_QUERY'
        }
    elif message in PROGRAMS.values():
        return {
            "text": f"Selected program: {message} What's your full name?",
            "options": [{"value": "menu", "label": "Return to Menu"}]
        }
    else:
        return {
            "text": "Invalid program selection. Please choose a program:",
            "options": [
                {"value": v, "label": v} for k, v in PROGRAMS.items()
            ],
            "new_state": 'PROGRAM_SELECTION'
        }



def handle_program_info(message):
    """ Handles the display of program information based on user choice """
    program = PROGRAMS.get(message)
    if program:
        details = {
            'Kids Program': "👶 Kids Program (4-14 years)\n- 8 skill levels\n- Certified instructors",
            'Adults Program': "🏊 Adults Program\n- Beginner to advanced\n- Flexible scheduling",
            'Ladies-Only Aqua Fitness': "🚺 Ladies-Only Aqua Fitness\n- Women-only sessions\n- Full-body workout",
            'Baby & Toddler Program': "👶👨👩 Baby & Toddler\n- Parent-child classes\n- Water safety basics",
            'Special Needs Program': "🌟 Special Needs Program\n- Adapted curriculum\n- Individual attention"
        }.get(message, "Program details not available")
        
        return {
            "text": f"{program} Details:\n{details}",
            "options": [{"value": "menu", "label": "Return to Menu"}],
            "new_state": "MAIN_MENU"
        }
    else:
        return {
            "text": "Invalid choice. Please select a program:",
            "options": [
                {"value": str(k), "label": v} for k, v in PROGRAMS.items()
            ],
            "new_state": 'PROGRAM_INFO'
        }


def handle_program_selection(message):
    """ Prepares for booking by capturing the selected program """

    program = PROGRAMS.get(message)
    if not program and message in PROGRAMS.values():  
        program = message

    if program:
        session['booking_data'] = {'program': program}
        session['booking_step'] = 'GET_NAME'
        return {
            "text": f"Selected program: {program}\nWhat's your full name?",
            "new_state": 'BOOKING_PROGRAM'
        }
    
    return {
        "text": "Invalid program selection. Please choose a program:",
        "options": [{"value": v, "label": v} for v in PROGRAMS.values()],
        "new_state": 'PROGRAM_SELECTION'
    }


def handle_booking(message: str) -> dict:
    """Handle individual booking steps"""
    try:
        current_step = session.get('booking_step')
        booking_data = session.get('booking_data', {})
        
        if not current_step:
            return {"text": "⚠️ Booking session expired. Please start over.", "new_state": 'MAIN_MENU'}
        
        field = current_step.split('_')[1].lower()
        booking_data[field] = message
        session['booking_data'] = booking_data
        
        next_missing = get_next_missing_field(booking_data)
        
        if next_missing:
            
            session['booking_step'] = f'GET_{next_missing.upper()}'
            prompts = {
                'program': "Which program would you like to join?",
                'name': "What's your full name?",
                'phone': "📱 What's your phone number?",
                'email': "📧 What's your email address?"
            }
            return {"text": prompts[next_missing]}

        else:

            booking_data['timestamp'] = datetime.now().isoformat()
            save_inquiry(booking_data)
            
            confirmation = (
                "✅ Booking confirmed!\n"
                f"Program: {booking_data['program']}\n"
                f"Name: {booking_data['name']}\n"
                f"Phone: {booking_data['phone']}\n"
                f"Email: {booking_data['email']}\n\n"
                "We'll contact you soon!"
            )
            
            session.pop('booking_data', None)
            session.pop('booking_step', None)
            
            return {
                "text": confirmation,
                "options": [{"value": "menu", "label": "Return to Menu"}],
                "new_state": 'MAIN_MENU'
            }
            
    except Exception as e:
        logger.error(f"Booking error: {str(e)}")
        return {"text": "⚠️ Booking failed. Type 'menu' to restart."}


def get_next_missing_field(booking_data: dict) -> Optional[str]:
    """Return the next missing required field"""
    required_fields = ['program', 'name', 'phone', 'email']
    for field in required_fields:
        if not booking_data.get(field):
            return field
    return None

def handle_ai_query(message: str) -> dict:
    """Handles AI queries and ensures all booking details are collected step-by-step."""

    try:
        # Retrieve or initialize booking data in session
        booking_data = session.get('booking_data', {})

        # Extract new details from user input
        booking_info = extract_booking_info(message)
        booking_info_dict = booking_info.dict()

        # Update session with newly extracted fields
        for key, value in booking_info_dict.items():
            if value:  # Only update if a value is extracted (not None)
                booking_data[key] = value

        # Save updated booking data in session
        session['booking_data'] = booking_data

        # Check for any missing fields
        missing_fields = [field for field in ['program', 'name', 'phone', 'email'] if field not in booking_data or booking_data[field] is None]

        if missing_fields:
            # Generate confirmation text for collected info
            confirmed_info = "\n".join([f"✅ {k.capitalize()}: {v}" for k, v in booking_data.items() if v])

            # Ask for the next missing field
            next_missing = missing_fields[0]
            field_prompts = {
                "program": "Which program would you like to join?",
                "name": "What's your full name?",
                "phone": "📱 What's your phone number?",
                "email": "📧 What's your email address?"
            }

            response_text = "Let's get your booking details.\n\n"
            if confirmed_info:
                response_text += f"I have the following details:\n{confirmed_info}\n\n"
            response_text += f"Please provide: {field_prompts[next_missing]}"

            return {"text": response_text, "new_state": f'BOOKING_{next_missing.upper()}'}

        # ✅ All fields collected → Confirm Booking
        booking_data['timestamp'] = datetime.now().isoformat()
        save_inquiry(booking_data)  # Save to database

        confirmation = (
            "✅ Booking confirmed!\n"
            f"Program: {booking_data['program']}\n"
            f"Name: {booking_data['name']}\n"
            f"Phone: {booking_data['phone']}\n"
            f"Email: {booking_data['email']}\n\n"
            "We'll contact you soon!"
        )

        # Clear session after successful booking
        session.pop('booking_data', None)

        return {"text": confirmation, "options": [{"value": "menu", "label": "Return to Menu"}], "new_state": 'MAIN_MENU'}

    except Exception as e:
        logger.error(f"AI Query Failed: {e}")
        return {"text": "Our AI agent is currently busy. Please try again later."}






if __name__ == '__main__':
    app.run(port=5000)