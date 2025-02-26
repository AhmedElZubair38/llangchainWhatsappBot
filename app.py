import re
import os
import uuid
import requests
import logging
import threading
from dotenv import load_dotenv
from datetime import datetime
from pydantic import BaseModel, Field
from rapidfuzz.process import extractOne
import rapidfuzz.process as rf_process
from fuzzywuzzy import process
from typing import Optional, Dict, List
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

API_BASE_URL = "http://localhost:5001"

@app.route('/send_inquiry', methods=['POST'])
def send_inquiry(data):
    """Send inquiry data to the API on port 5001"""

    logger.info(f"📌 Received from bot: {data}")

    response = requests.post(f"{API_BASE_URL}/add_inquiry", json=data)

    if response.status_code == 200:
        logger.info("Data successfully sent to API.")
        return jsonify({"message": "Inquiry sent to API!"}), 200
    else:
        logger.error(f"Failed to send data to API: {response.status_code}")
        return jsonify({"error": "Failed to send inquiry"}), 500


@app.route('/fetch_inquiries', methods=['GET'])
def fetch_inquiries():
    """Fetch inquiries from the API on port 5001"""
    response = requests.get(f"{API_BASE_URL}/get_inquiries")

    if response.status_code == 200:
        inquiries = response.json()
        return jsonify(inquiries), 200
    else:
        logger.error(f"Failed to fetch inquiries: {response.status_code}")
        return jsonify({"error": "Failed to retrieve inquiries"}), 500


class BookingInfo(BaseModel):
    program: Optional[str] = Field(None, description="The swimming program name")
    name: Optional[str] = Field(None, description="Customer's full name")
    phone: Optional[str] = Field(None, description="Customer's phone number")
    email: Optional[str] = Field(None, description="Customer's email address")


def extract_booking_info(query: str) -> BookingInfo:
    """Extract booking information from a natural language query using LangChain"""

    llm = ChatOpenAI(
        model="deepseek-llm",
        base_url="http://172.27.240.1:11434/v1",
        temperature=0.1
    )
    
    parser = PydanticOutputParser(pydantic_object=BookingInfo)
    
    prompt = f"""
    Extract booking information from the following query. If information is not present, return null for that field.
    
    Query: {query}
    
    Extract these fields:
    - Program name (match to: Kids Program, Adults Program, Ladies-Only Aqua Fitness, Baby & Toddler Program, Special Needs Program), if they are unsure about its name or they have a typo, try being smart and figure out which program from the predefined list they mean.
    - Full name
    - Phone number
    - Email address
    
    {parser.get_format_instructions()}
    """
    
    messages = [
        SystemMessage(content="You are a helpful assistant that extracts booking information from text."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)

    return parser.parse(response.content)


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
            "text": "Ask me anything about our programs!",
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


# def handle_booking(message: str) -> dict:
#     """Handle individual booking steps"""

#     try:
#         current_step = session.get('booking_step')
#         booking_data = session.get('booking_data', {})
        
#         if not current_step:
#             return {"text": "⚠️ Booking session expired. Please start over.", "new_state": 'MAIN_MENU'}
        
#         field = current_step.split('_')[1].lower()
#         booking_data[field] = message
#         session['booking_data'] = booking_data
        
#         next_missing = get_next_missing_field(booking_data)
        
#         if next_missing:
            
#             session['booking_step'] = f'GET_{next_missing.upper()}'
#             prompts = {
#                 'program': "Which program would you like to join?",
#                 'name': "What's your full name?",
#                 'phone': "📱 What's your phone number?",
#                 'email': "📧 What's your email address?"
#             }

#             missing_field_prompts = {
#                     'program': "Could you please tell me which program you'd like to join?",
#                     'name': "Could you please provide your full name?",
#                     'phone': "Could you please share your phone number?",
#                     'email': "Could you please provide your email address?"
#             }

#             return {"text": missing_field_prompts[next_missing]}

#         else:

#             booking_data['timestamp'] = datetime.now().isoformat()
            
#             confirmation = (
#                 "✅ Booking confirmed!\n"
#                 f"Program: {booking_data['program']}\n"
#                 f"Name: {booking_data['name']}\n"
#                 f"Phone: {booking_data['phone']}\n"
#                 f"Email: {booking_data['email']}\n\n"
#                 "We'll contact you soon!"
#             )

#             send_inquiry(booking_data)
            
#             session.pop('booking_data', None)
#             session.pop('booking_step', None)
            
#             return {
#                 "text": confirmation,
#                 "options": [{"value": "menu", "label": "Return to Menu"}],
#                 "new_state": 'MAIN_MENU'
#             }
            
#     except Exception as e:
#         logger.error(f"Booking error: {str(e)}")
#         return {"text": "⚠️ Booking failed. Type 'menu' to restart."}



def handle_booking(message: str) -> dict:
    """Handle individual booking steps with input validation."""
    try:
        current_step = session.get('booking_step')
        booking_data = session.get('booking_data', {})

        if not current_step:
            return {"text": "⚠️ Booking session expired. Please start over.", "new_state": 'MAIN_MENU'}

        # Determine which field we are capturing
        field = current_step.split('_')[1].lower()

        # Validate input based on the field type:
        if field == "email":
            validated_email = extract_email(message)
            if not validated_email:
                # Re-prompt for valid email, keeping the same state
                return {"text": "⚠️ The email address provided is invalid. Please enter a valid email address:"}
            booking_data[field] = validated_email

        elif field == "phone":
            validated_phone = extract_phone(message)
            if not validated_phone:
                # Re-prompt for valid phone number
                return {"text": "⚠️ The phone number provided is invalid. Please enter a valid phone number:"}
            booking_data[field] = validated_phone

        elif field == "name":
            # Simple check: full name should contain at least two words
            if len(message.split()) < 2:
                return {"text": "⚠️ Please provide your full name (first and last name):"}
            booking_data[field] = message

        else:
            booking_data[field] = message

        session['booking_data'] = booking_data

        # Check for the next missing field
        next_missing = get_next_missing_field(booking_data)
        if next_missing:
            session['booking_step'] = f'GET_{next_missing.upper()}'
            missing_field_prompts = {
                'program': "Could you please tell me which program you'd like to join?",
                'name': "Could you please provide your full name?",
                'phone': "Could you please share your phone number?",
                'email': "Could you please provide your email address?"
            }
            return {"text": missing_field_prompts[next_missing]}
        else:
            booking_data['timestamp'] = datetime.now().isoformat()
            confirmation = (
                "✅ Booking confirmed!\n"
                f"Program: {booking_data['program']}\n"
                f"Name: {booking_data['name']}\n"
                f"Phone: {booking_data['phone']}\n"
                f"Email: {booking_data['email']}\n\n"
                "We'll contact you soon!"
            )
            send_inquiry(booking_data)
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




def extract_email(text: str) -> Optional[str]:
    """Extract email using regex pattern"""

    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, text)
    return matches[0] if matches else None

# def extract_phone(text: str) -> Optional[str]:
#     """Extract phone number using regex pattern"""

#     phone_pattern = r'(?:\+?\d{1,4}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}'
#     matches = re.findall(phone_pattern, text)
#     return matches[0] if matches else None

def extract_phone(text: str) -> Optional[str]:
    """Extract and normalize a UAE phone number to +971 format.
    
    Accepts:
    - International format: +971XXXXXXXXX (where XXXXXXXXX is 9 digits, total 13 characters)
    - Local format: 0XXXXXXXXX (10 digits total)
    
    Transforms local format into international format by replacing the leading 0 with +971.
    Returns the phone number in +971XXXXXXXXX format, or None if the input is invalid.
    """
    text = text.strip()
    
    if text.startswith('+'):
        # Remove any non-digit characters after the plus sign.
        phone = '+' + re.sub(r'\D', '', text[1:])
        # Validate: must start with +971 and have exactly 13 characters (+971 + 9 digits)
        if phone.startswith("+971") and len(phone) == 13:
            return phone
        else:
            return None
    else:
        # Remove non-digit characters.
        phone = re.sub(r'\D', '', text)
        # Validate local format: must start with 0 and be exactly 10 digits.
        if phone.startswith("0") and len(phone) == 10:
            return "+971" + phone[1:]
        else:
            return None


def extract_program(text: str) -> Optional[str]:
    """Extract program using keyword-based matching first, then RapidFuzz."""

    if not text:
        return None

    text_lower = text.lower().strip()

    keyword_mapping = {
        "kids": "Kids Program",
        "adults": "Adults Program",
        "ladies": "Ladies-Only Aqua Fitness",
        "baby": "Baby & Toddler Program",
        "toddler": "Baby & Toddler Program",
        "special needs": "Special Needs Program"
    }

    for keyword, program in keyword_mapping.items():
        if keyword in text_lower:
            return program

    programs = [program.lower().strip() for program in PROGRAMS.values()]
    best_match = rf_process.extractOne(text_lower, programs, score_cutoff=90)

    if best_match:
        matched_program = best_match[0]
        for original_program in PROGRAMS.values():
            if original_program.lower().strip() == matched_program:
                return original_program

    return None

def extract_name(text: str) -> Optional[str]:
    """Extract full name from text with enhanced flexibility for various name expressions."""

    name_patterns = [
        r'(?:my\sname\sis\s)([A-Z][a-zA-Z\'\-]+(?:\s[A-Z][a-zA-Z\'\-]+)?)',  # "My name is"
        r'(?:his\sname\sis\s)([A-Z][a-zA-Z\'\-]+(?:\s[A-Z][a-zA-Z\'\-]+)?)',  # "His name is"
        r'(?:her\sname\sis\s)([A-Z][a-zA-Z\'\-]+(?:\s[A-Z][a-zA-Z\'\-]+)?)',  # "Her name is"
        r'(?:i\'?m\s)([A-Z][a-zA-Z\'\-]+(?:\s[A-Z][a-zA-Z\'\-]+)?)',  # "I'm"
        r'(?:i\sam\s)([A-Z][a-zA-Z\'\-]+(?:\s[A-Z][a-zA-Z\'\-]+)?)',  # "I am"
        r'(?:this\sis\s)([A-Z][a-zA-Z\'\-]+(?:\s[A-Z][a-zA-Z\'\-]+)?)',  # "This is"
        r'(?:call\sme\s)([A-Z][a-zA-Z\'\-]+(?:\s[A-Z][a-zA-Z\'\-]+)?)',  # "Call me"
        r'(?:i\s\'?m\scalled\s)([A-Z][a-zA-Z\'\-]+(?:\s[A-Z][a-zA-Z\'\-]+)?)',  # "I'm called"
        r'(?:you\scan\scall\sme\s)([A-Z][a-zA-Z\'\-]+(?:\s[A-Z][a-zA-Z\'\-]+)?)',  # "You can call me"
        r'(?:my\sfriends\scall\sme\s)([A-Z][a-zA-Z\'\-]+(?:\s[A-Z][a-zA-Z\'\-]+)?)',  # "My friends call me"
        r'(?:it\'s\s)([A-Z][a-zA-Z\'\-]+(?:\s[A-Z][a-zA-Z\'\-]+)?)',  # "It's"
    ]

    for pattern in name_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[0]
    
    return None

def get_next_missing_field(booking_data: dict) -> Optional[str]:
    """Return the next missing required field"""

    required_fields = ['program', 'name', 'phone', 'email']
    for field in required_fields:
        if not booking_data.get(field):
            return field
    return None

def handle_ai_query(message: str) -> dict:
    """Enhanced AI query handler with booking capabilities"""
    try:

        session.pop('booking_data', None)
        session.pop('booking_step', None)

        booking_keywords = ['book', 'register', 'sign up', 'enroll', 'join']
        is_booking_request = any(keyword in message.lower() for keyword in booking_keywords)
        
        if is_booking_request:

            extracted_data = {
                'email': extract_email(message),
                'phone': extract_phone(message),
                'program': extract_program(message),
                'name': extract_name(message)
            }
            
            booking_data = session.get('booking_data', {})
            
            booking_data.update({k: v for k, v in extracted_data.items() if v is not None})
            
            session['booking_data'] = booking_data
            
            next_missing = get_next_missing_field(booking_data)
            
            if next_missing:

                confirmed_info = []

                if booking_data.get('program'):
                    confirmed_info.append(f"Program: {booking_data['program']}")
                if booking_data.get('name'):
                    confirmed_info.append(f"Name: {booking_data['name']}")
                if booking_data.get('phone'):
                    confirmed_info.append(f"Phone: {booking_data['phone']}")
                if booking_data.get('email'):
                    confirmed_info.append(f"Email: {booking_data['email']}")
                
                info_text = "\n".join(confirmed_info) if confirmed_info else ""
                
                session['booking_step'] = f'GET_{next_missing.upper()}'
                
                prompts = {
                    'program': "Which program would you like to join?",
                    'name': "What's your full name?",
                    'phone': "📱 What's your phone number?",
                    'email': "📧 What's your email address?"
                }

                missing_field_prompts = {
                    'program': "Could you please tell me which program you'd like to join?",
                    'name': "Could you please provide your full name?",
                    'phone': "Could you please share your phone number?",
                    'email': "Could you please provide your email address?"
                }
                
                response_text = "Sure, I'd be happy to help you with the booking! 😊\n\n"

                if info_text:
                    response_text += f"Here’s what I’ve gathered so far:\n{info_text}\n\n"
                    
                response_text += (
                    "To proceed, "
                    f"{missing_field_prompts[next_missing].lower()}"
                )
                
                if next_missing == 'program':
                    return {
                        "text": response_text,
                        "options": [{"value": k, "label": v} for k, v in PROGRAMS.items()],
                        "new_state": 'BOOKING_PROGRAM'
                    }
                else:
                    return {
                        "text": response_text,
                        "new_state": 'BOOKING_PROGRAM'
                    }
            else:

                booking_data['timestamp'] = datetime.now().isoformat()
                send_inquiry(booking_data)
                
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
            
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="chroma_db"
        )
        retriever = vector_store.as_retriever(search_kwargs={'k': 100})
        docs = retriever.invoke(message)
        knowledge = "\n\n".join([doc.page_content.strip() for doc in docs])
        
        llm = ChatOpenAI(
            model="deepseek-llm",
            base_url="http://172.27.240.1:11434/v1",
            temperature=0.1
        )
        
        messages = [SystemMessage(content=f"""You're an expert assistant for Aquasprint Swimming Academy. Follow these rules:
            1. Answer ONLY using the knowledge base fed to you.
            2. Be concise and professional.
            3. If unsure or outside of your knowledge scope and role, YOU HAVE TO SAY "I don't have that information".
            4. Never make up answers.
            5. If the user shows interest in booking or classes, remind them they can book directly by saying something like 
               "Would you like to book a class? Just tell me your preferred program and contact details!
            6. If a user asks you a question which requieres explanation or a lot of talking...just be concise and dont yap
            7. Yes you are the Aquasprint Swimming Academy bot  or agent...thats why when the user talks to you about the academy, you are supposed to answer as in like "we" becuase you are part of us! you are part of Aquasprint Swimming Academy"

            Knowledge Base:
            {knowledge}"""),
            HumanMessage(content=message)
        ]
        
        ai_response = llm.invoke(messages)
        return {
            "text": f"🤖 AI Agent:\n{ai_response.content}",
            "options": [{"value": "menu", "label": "Return to Menu"}]
        }
    
    except Exception as e:
        logger.error(f"AI Query Failed: {e}")
        return {"text": "Our AI agent is currently busy. Please try again later."}


if __name__ == '__main__':
    app.run(port=5000)