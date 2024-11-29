import streamlit as st
import pandas as pd
import os

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

# Custom CSS for profile icon and alignment
st.markdown("""
    <style>
    .profile-icon {
        position: fixed;
        top: 65px;
        right: 80px;
        z-index: 1000;
        cursor: pointer;
    }
    .profile-circle {
        border-radius: 50%;
        background-color: #4CAF50;
        color: white;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: bold;
    }
    .profile-toggle-button {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# Main section content
if 'login_successful' not in st.session_state or not st.session_state.login_successful:
    # Login / Signup page content
    st.markdown("<h1 style='text-align: center;'>Welcome to the Test Case Generator</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        The **Test Case Generator** allows you to create and manage test cases for your projects. 
        You can upload files such as **videos, screenshots**, and **documents** and associate them with detailed test case descriptions.
        
        To get started, please either log in or sign up if you don't have an account yet.
        """
    )

    login_or_signup = st.radio("Choose action:", ["Login", "Sign Up"], index=0)

    if login_or_signup == "Login":
        user_email = st.text_input("Email")
        user_password = st.text_input("Password", type="password")
        login_button = st.button("Login")

        if login_button:
            if check_credentials(user_email, user_password):
                st.session_state.login_successful = True  # Set login success state
                st.session_state.user_email = user_email  # Store email in session state
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
        # Dropdown for Test Case Type
        test_case_type = st.selectbox(
            "Test Case Type",
            ["Test Case Generation", "Test Case Validation", "Context Modeling"],
            help="Choose the type of test case you are working on."
        )

    with col2:
        # Dropdown for Content Type
        content_type = st.selectbox(
            "Content Type",
            ["Photo", "Video", "Document"],
            help="Choose the type of content for this test case."
        )

    with col3:
        # File upload based on content type with unique keys to avoid duplicate ID errors
        uploaded_file = st.file_uploader("Upload File", type=["png", "jpg", "jpeg", "mp4", "mov", "avi", "pdf", "docx", "txt"])

    # Submit button for saving the test case
    if st.button("Submit Test Case"):
        if not uploaded_file:
            st.warning("Please upload a file for the selected content type.")
        else:
            save_data(st.session_state.user_email, test_case_type, uploaded_file.name)
            st.success(f"Test case with file '{uploaded_file.name}' saved successfully!")
