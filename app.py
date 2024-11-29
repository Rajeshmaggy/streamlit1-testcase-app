import streamlit as st
import pandas as pd
import os

# Set up the page
st.set_page_config(page_title="Test Case Generator", layout="wide")

# File path to store data
DATA_FILE = "test_cases.csv"
USER_CREDENTIALS_FILE = "user_credentials.csv"

# Initialize session state
if 'login_successful' not in st.session_state:
    st.session_state.login_successful = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""

# Ensure data files exist
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=["Email", "Test Case Type", "File Name"]).to_csv(DATA_FILE, index=False)

if not os.path.exists(USER_CREDENTIALS_FILE):
    pd.DataFrame(columns=["Email", "Password"]).to_csv(USER_CREDENTIALS_FILE, index=False)

# Load data files
try:
    test_cases_df = pd.read_csv(DATA_FILE)
    user_credentials_df = pd.read_csv(USER_CREDENTIALS_FILE)
except Exception as e:
    st.error("Error loading data files. Please check your CSV files.")
    st.stop()

# Function to save user credentials
def save_user_credentials(email, password):
    global user_credentials_df
    new_entry = pd.DataFrame({"Email": [email], "Password": [password]})
    user_credentials_df = pd.concat([user_credentials_df, new_entry], ignore_index=True)
    user_credentials_df.to_csv(USER_CREDENTIALS_FILE, index=False)

# Function to check credentials
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

# Sidebar for user history
if st.session_state.login_successful:
    user_history = test_cases_df[test_cases_df["Email"] == st.session_state.user_email]

    with st.sidebar:
        st.markdown("### User History")
        if user_history.empty:
            st.info("No test cases submitted yet.")
        else:
            filter_type = st.selectbox(
                "Filter by Test Case Type:",
                ["All"] + user_history["Test Case Type"].unique().tolist()
            )
            if filter_type != "All":
                filtered_history = user_history[user_history["Test Case Type"] == filter_type]
            else:
                filtered_history = user_history
            st.dataframe(filtered_history[["Test Case Type", "File Name"]].reset_index(drop=True))

# Login or Signup Page
if not st.session_state.login_successful:
    st.title("Welcome to the Test Case Generator")

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

# Main Test Case Generator (after login)
else:
    st.title("Test Case Generator")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        test_case_type = st.selectbox(
            "Test Case Type",
            ["Test Case Generation", "Test Case Validation", "Context Modeling"]
        )

    with col2:
        content_type = None
        if test_case_type == "Test Case Generation":
            content_type = st.selectbox(
                "Content Type", ["Photo", "Video", "Document"]
            )

    with col3:
        uploaded_file = None
        if content_type == "Photo":
            uploaded_file = st.file_uploader("Upload Photo", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        elif content_type == "Video":
            uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
        elif content_type == "Document":
            uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])

    if st.button("Submit Test Case"):
        if not uploaded_file:
            st.warning("Please upload a file for the selected content type.")
        else:
            if isinstance(uploaded_file, list):
                for file in uploaded_file:
                    save_data(st.session_state.user_email, test_case_type, file.name)
                st.success(f"{len(uploaded_file)} files uploaded successfully!")
            else:
                save_data(st.session_state.user_email, test_case_type, uploaded_file.name)
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
