import streamlit as st
import pandas as pd
import os
import cv2
import pytesseract
from fpdf import FPDF
import tempfile

# Set up the page
st.set_page_config(page_title="Test Case Generator", layout="wide")

# File path to store data
DATA_FILE = "test_cases.csv"
USER_CREDENTIALS_FILE = "user_credentials.csv"

# Load existing data or create a new DataFrame
if os.path.exists(DATA_FILE):
    test_cases_df = pd.read_csv(DATA_FILE)
else:
    test_cases_df = pd.DataFrame(columns=["Email", "Test Case Type", "File Name"])

# Load user credentials from the file
if os.path.exists(USER_CREDENTIALS_FILE):
    user_credentials_df = pd.read_csv(USER_CREDENTIALS_FILE)
else:
    user_credentials_df = pd.DataFrame(columns=["Email", "Password"])

# Function to save user credentials
def save_user_credentials(email, password):
    global user_credentials_df
    new_entry = pd.DataFrame({"Email": [email], "Password": [password]})
    user_credentials_df = pd.concat([user_credentials_df, new_entry], ignore_index=True)
    user_credentials_df.to_csv(USER_CREDENTIALS_FILE, index=False)

# Function to check if credentials are valid
def check_credentials(email, password):
    user = user_credentials_df[user_credentials_df["Email"] == email]
    if not user.empty and user["Password"].values[0] == password:
        return True
    return False

# Function to save test case data
def save_data(email, test_case_type, file_name):
    global test_cases_df
    new_entry = pd.DataFrame(
        {"Email": [email], "Test Case Type": [test_case_type], "File Name": [file_name]}
    )
    test_cases_df = pd.concat([test_cases_df, new_entry], ignore_index=True)
    test_cases_df.to_csv(DATA_FILE, index=False)

# OCR and Video Processing Logic
def extract_frames(video_path, interval=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if frame_count % (interval * 30) == 0:  # Capture frame every `interval` seconds
            if success:
                frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def extract_text_from_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_frame)
    return text

def generate_prd_from_text(extracted_text):
    # Simulating PRD generation with a placeholder response
    return f"Generated PRD based on the following extracted text:\n\n{extracted_text}"

def save_text_to_pdf(text, output_pdf_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in text.split('\n'):
        pdf.multi_cell(0, 10, line)

    pdf.output(output_pdf_path)

# Main section content
if 'login_successful' not in st.session_state or not st.session_state.login_successful:
    # Login / Signup page content
    st.markdown("<h1 style='text-align: center;'>Welcome to the Test Case Generator</h1>", unsafe_allow_html=True)

    login_or_signup = st.radio("Choose action:", ["Login", "Sign Up"], index=0)

    if login_or_signup == "Login":
        user_email = st.text_input("Email")
        user_password = st.text_input("Password", type="password")
        login_button = st.button("Login")

        if login_button:
            if check_credentials(user_email, user_password):
                st.session_state.login_successful = True
                st.session_state.user_email = user_email
                st.success("Login successful!")
            else:
                st.error("Invalid credentials. Please try again.")

    elif login_or_signup == "Sign Up":
        new_user_email = st.text_input("Email")
        new_user_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        signup_button = st.button("Sign Up")

        if signup_button:
            if new_user_password != confirm_password:
                st.error("Passwords do not match. Please try again.")
            elif new_user_email in user_credentials_df["Email"].values:
                st.error("Email already exists. Please log in.")
            else:
                save_user_credentials(new_user_email, new_user_password)
                st.success("Signup successful! You can now log in.")
else:
    # Sidebar - User History
    with st.sidebar:
        st.markdown("### Test Case History")
        user_history = test_cases_df[test_cases_df["Email"] == st.session_state.user_email]
        if user_history.empty:
            st.info("No test cases submitted yet.")
        else:
            for _, row in user_history.iterrows():
                st.markdown(f"- **{row['Test Case Type']}**: {row['File Name']}")

    # Main Section Content
    st.markdown("<h1 style='text-align: center;'>Test Case Generator</h1>", unsafe_allow_html=True)

    # Test Case Input Section
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        test_case_type = st.selectbox(
            "Test Case Type",
            ["Test Case Generation", "Test Case Validation", "Context Modeling"]
        )

    with col2:
        content_type = st.selectbox(
            "Content Type",
            ["Photo", "Video", "Document"]
        )

    with col3:
        uploaded_file = st.file_uploader("Upload File", type=["mp4", "mov", "avi"])

    if st.button("Submit Test Case"):
        if not uploaded_file:
            st.warning("Please upload a file.")
        elif test_case_type == "Test Case Generation" and content_type == "Video":
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name

            frames = extract_frames(video_path, interval=2)
            extracted_text = "".join(
                [f"--- Frame {idx} ---\n{extract_text_from_frame(frame)}\n" for idx, frame in enumerate(frames)]
            )

            if extracted_text.strip():
                prd = generate_prd_from_text(extracted_text)
                save_path = "prd_document.pdf"
                save_text_to_pdf(prd, save_path)
                st.success("PRD document generated successfully!")
                with open(save_path, "rb") as pdf_file:
                    st.download_button("Download PRD", data=pdf_file, file_name="prd_document.pdf")
            else:
                st.warning("No text detected in the video.")
