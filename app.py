import streamlit as st
import pandas as pd
import os

# Set up the page
st.set_page_config(page_title="Test Case Generator", layout="wide")

# File paths to store data
DATA_FILE = "test_cases.csv"
USERS_FILE = "users.csv"

# Load existing test case data or create a new DataFrame
if os.path.exists(DATA_FILE):
    test_cases_df = pd.read_csv(DATA_FILE)
else:
    test_cases_df = pd.DataFrame(columns=["Email", "Test Case Type", "Test Case Details", "File Name"])

# Load or create users data
if os.path.exists(USERS_FILE):
    users_df = pd.read_csv(USERS_FILE)
else:
    users_df = pd.DataFrame(columns=["Email", "Password"])

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

# Function to add a new user
def add_user(email, password):
    global users_df
    new_user = pd.DataFrame({"Email": [email], "Password": [password]})
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv(USERS_FILE, index=False)

# Function to verify login credentials
def verify_login(email, password):
    global users_df
    return any((users_df["Email"] == email) & (users_df["Password"] == password))

# Login/Register logic
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center;'>Login Page</h1>", unsafe_allow_html=True)
    login_option = st.radio("Choose an option:", ["Login", "Register"])

    if login_option == "Login":
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if verify_login(email, password):
                st.success("Login successful!")
                st.session_state.logged_in = True
                st.session_state.user_email = email
            else:
                st.error("Invalid email or password.")
    elif login_option == "Register":
        email = st.text_input("Email (Registration)")
        password = st.text_input("Password (Registration)", type="password")
        if st.button("Register"):
            if email in users_df["Email"].values:
                st.error("This email is already registered.")
            else:
                add_user(email, password)
                st.success("Registration successful! Please login.")
else:
    # Sidebar: User email input (Already logged in)
    user_email = st.session_state.user_email
    st.sidebar.header("User Info")
    st.sidebar.write(f"Logged in as: **{user_email}**")

    # Display previously submitted test cases in the sidebar
    user_test_cases = test_cases_df[test_cases_df["Email"] == user_email]
    if not user_test_cases.empty:
        st.sidebar.markdown("### Your Previously Submitted Test Cases:")
        for i, row in user_test_cases.iterrows():
            st.sidebar.write(f"- **Type**: {row['Test Case Type']}, **Details**: {row['Test Case Details']}")
    else:
        st.sidebar.info("No previous test cases found for this email.")

    # Main section
    st.markdown("<h1 style='text-align: center;'>Test Case Generator</h1>", unsafe_allow_html=True)

    # Dropdown to select input type
    test_case_type = st.selectbox(
        "Select Test Case Type:",
        ["Video", "Screenshots", "Document"],
        help="Choose the type of test case you are performing.",
    )

    # # Test case details input
    # test_case_details = st.text_area(
    #     "Enter Test Case Details:",
    #     help="Provide details about the test case you are performing.",
    # )

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
        if not test_case_details.strip():
            st.warning("Please provide test case details.")
        elif not uploaded_file:
            st.warning("Please upload a file for the selected test case type.")
        else:
            # Handle multiple screenshots case
            if isinstance(uploaded_file, list):  # For multiple files (screenshots)
                for file in uploaded_file:
                    save_data(user_email, test_case_type, test_case_details, file.name)
                st.success(f"{len(uploaded_file)} screenshots uploaded and test case saved successfully!")
            else:  # For single file (video/document)
                save_data(user_email, test_case_type, test_case_details, uploaded_file.name)
                st.success(f"Test case with file '{uploaded_file.name}' saved successfully!")

            # Reload the user test cases
            user_test_cases = test_cases_df[test_cases_df["Email"] == user_email]

            # Display the updated test cases in the sidebar
            st.sidebar.markdown("### Your Updated Test Cases:")
            for i, row in user_test_cases.iterrows():
                st.sidebar.write(f"- **Type**: {row['Test Case Type']}, **Details**: {row['Test Case Details']}")
