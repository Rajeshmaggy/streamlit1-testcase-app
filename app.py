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
    test_cases_df = pd.DataFrame(columns=["Email", "Test Case Type", "Test Case Details", "File Name"])

# Load user credentials from the file
if os.path.exists(USER_CREDENTIALS_FILE):
    user_credentials_df = pd.read_csv(USER_CREDENTIALS_FILE)
else:
    user_credentials_df = pd.DataFrame(columns=["Email", "Password"])

# Function to save user credentials
def save_user_credentials(email, password):
    global user_credentials_df
    new_entry = pd.DataFrame(
        {"Email": [email], "Password": [password]}
    )
    user_credentials_df = pd.concat([user_credentials_df, new_entry], ignore_index=True)
    user_credentials_df.to_csv(USER_CREDENTIALS_FILE, index=False)

# Function to check if credentials are valid
def check_credentials(email, password):
    user = user_credentials_df[user_credentials_df["Email"] == email]
    if not user.empty and user["Password"].values[0] == password:
        return True
    return False

# Function to save test case data
def save_data(email, test_case_type, test_case_details, file_name):
    global test_cases_df
    new_entry = pd.DataFrame(
        {
            "Email": [email],
            "Test Case Type": [test_case_type],
            "Test Case Details": [test_case_details],
            "File Name": [file_name],
        }
    )
    test_cases_df = pd.concat([test_cases_df, new_entry], ignore_index=True)
    test_cases_df.to_csv(DATA_FILE, index=False)

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
    # Main section - Test Case Generator (Only visible after login)
    st.markdown("<h1 style='text-align: center;'>Test Case Generator</h1>", unsafe_allow_html=True)

    # Extract first letter of email for profile picture
    profile_initial = st.session_state.user_email[0].upper()

    # Profile Display in a Circular Badge
    st.markdown(
        f"""
        <style>
            .profile-badge {{
                position: fixed;
                top: 70px;
                right: 20px;
                width: 40px;
                height: 40px;
                background-color: #4CAF50;
                color: white;
                border-radius: 50%;
                text-align: center;
                font-size: 20px;
                font-weight: bold;
                line-height: 40px;
            }}
        </style>
        <div class="profile-badge">{profile_initial}</div>
        """,
        unsafe_allow_html=True
    )

    # Dropdown to select input type
    test_case_type = st.selectbox(
        "Select Test Case Type:",
        ["Video", "Screenshots", "Document"],
        help="Choose the type of test case you are performing.",
    )

    # Test case details input
    test_case_details = st.text_area(
        "Enter Test Case Details:",
        help="Provide details about the test case you are performing.",
    )

    # File upload section
    uploaded_file = None
    if test_case_type == "Video":
        uploaded_file = st.file_uploader("Upload your video file:", type=["mp4", "mov", "avi"])
    elif test_case_type == "Screenshots":
        uploaded_file = st.file_uploader("Upload your screenshots:", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    elif test_case_type == "Document":
        uploaded_file = st.file_uploader("Upload your document:", type=["pdf", "docx", "txt"])

    # Submit button
    if st.button("Submit Test Case"):
        if 'user_email' not in st.session_state:
            st.warning("Please log in to submit a test case.")
        elif not test_case_details.strip():
            st.warning("Please provide test case details.")
        elif not uploaded_file:
            st.warning("Please upload a file for the selected test case type.")
        else:
            # Handle multiple screenshots case
            if isinstance(uploaded_file, list):  # For multiple files (screenshots)
                for file in uploaded_file:
                    save_data(st.session_state.user_email, test_case_type, test_case_details, file.name)
                st.success(f"{len(uploaded_file)} screenshots uploaded and test case saved successfully!")
            else:  # For single file (video/document)
                save_data(st.session_state.user_email, test_case_type, test_case_details, uploaded_file.name)
                st.success(f"Test case with file '{uploaded_file.name}' saved successfully!")

            # Reload the user test cases
            user_test_cases = test_cases_df[test_cases_df["Email"] == st.session_state.user_email]

            # Display the updated test cases in the sidebar
            st.sidebar.markdown("### Your Updated Test Cases:")
            for i, row in user_test_cases.iterrows():
                st.sidebar.write(f"- **Type**: {row['Test Case Type']}, **Details**: {row['Test Case Details']}")
