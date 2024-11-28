import streamlit as st
import pandas as pd
import os

# Set up the page
st.set_page_config(page_title="Test Case Generator", layout="wide")

# File paths for storing data
DATA_FILE = "test_cases.csv"
USERS_FILE = "users.csv"

# Load test case data or create a new DataFrame
if os.path.exists(DATA_FILE):
    test_cases_df = pd.read_csv(DATA_FILE)
else:
    test_cases_df = pd.DataFrame(columns=["Email", "Test Case Type", "Test Case Details", "File Name"])

# Load or create user data
if os.path.exists(USERS_FILE):
    users_df = pd.read_csv(USERS_FILE)
else:
    users_df = pd.DataFrame(columns=["Email", "Password"])

# Functions for data handling
def save_test_case(email, test_case_type, test_case_details, file_name):
    """Save test case details."""
    global test_cases_df
    new_entry = pd.DataFrame({
        "Email": [email],
        "Test Case Type": [test_case_type],
        "Test Case Details": [test_case_details],
        "File Name": [file_name],
    })
    test_cases_df = pd.concat([test_cases_df, new_entry], ignore_index=True)
    test_cases_df.to_csv(DATA_FILE, index=False)

def add_user(email, password):
    """Add a new user."""
    global users_df
    new_user = pd.DataFrame({"Email": [email], "Password": [password]})
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv(USERS_FILE, index=False)

def verify_login(email, password):
    """Verify user login credentials."""
    global users_df
    return any((users_df["Email"] == email) & (users_df["Password"] == password))

# Initialize session states
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "show_modal" not in st.session_state:
    st.session_state.show_modal = False

# Top-right corner button
st.markdown(
    """
    <style>
        .top-right-button {
            position: absolute;
            top: 20px;
            right: 20px;
            background-color: #007BFF;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 10px 15px;
            font-size: 14px;
            cursor: pointer;
        }
        .top-right-button:hover {
            background-color: #0056b3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state.logged_in:
    st.markdown(
        f"<button class='top-right-button'>Logged in as {st.session_state.user_email}</button>",
        unsafe_allow_html=True,
    )
else:
    if st.button("Login/Signup", key="open_modal"):
        st.session_state.show_modal = True

# Modal popup for authentication
if st.session_state.show_modal:
    st.markdown(
        """
        <div style="
            position: fixed;
            top: 20%;
            left: 50%;
            transform: translate(-50%, -20%);
            width: 400px;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            z-index: 10;">
            <h3 style="text-align: center; margin-bottom: 20px;">User Authentication</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        login_option = st.radio("Choose an option:", ["Login", "Signup"], key="auth_option")
        email = st.text_input("Email", key="auth_email")
        password = st.text_input("Password", type="password", key="auth_password")

        if login_option == "Login" and st.button("Login", key="login_button"):
            if verify_login(email, password):
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.show_modal = False
                st.experimental_rerun()
            else:
                st.error("Invalid email or password.")
        elif login_option == "Signup" and st.button("Signup", key="signup_button"):
            if email in users_df["Email"].values:
                st.error("This email is already registered.")
            else:
                add_user(email, password)
                st.success("Signup successful! Please login.")
                st.session_state.show_modal = False
                st.experimental_rerun()

        if st.button("Close", key="close_modal"):
            st.session_state.show_modal = False
            st.experimental_rerun()

    # Dim the background while the modal is open
    st.markdown(
        """
        <style>
            .stApp {
                filter: blur(2px);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Main Test Case Generator content
if st.session_state.logged_in:
    st.sidebar.header("User Info")
    st.sidebar.write(f"Logged in as: **{st.session_state.user_email}**")

    # Display previous test cases in the sidebar
    user_test_cases = test_cases_df[test_cases_df["Email"] == st.session_state.user_email]
    if not user_test_cases.empty:
        st.sidebar.markdown("### Your Previously Submitted Test Cases:")
        for i, row in user_test_cases.iterrows():
            st.sidebar.write(f"- **Type**: {row['Test Case Type']}, **Details**: {row['Test Case Details']}")
    else:
        st.sidebar.info("No previous test cases found for this email.")

    # Main section
    st.markdown("<h1 style='text-align: center;'>Test Case Generator</h1>", unsafe_allow_html=True)

    # Dropdown to select test case type
    test_case_type = st.selectbox(
        "Select Test Case Type:",
        ["Video", "Screenshots", "Document"],
        help="Choose the type of test case you are performing.",
    )

    uploaded_file = None
    if test_case_type == "Video":
        uploaded_file = st.file_uploader("Upload your video file:", type=["mp4", "mov", "avi"])
    elif test_case_type == "Screenshots":
        uploaded_file = st.file_uploader("Upload your screenshots:", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    elif test_case_type == "Document":
        uploaded_file = st.file_uploader("Upload your document:", type=["pdf", "docx", "txt"])

    if st.button("Submit Test Case"):
        if uploaded_file:
            if isinstance(uploaded_file, list):  # Multiple screenshots
                for file in uploaded_file:
                    save_test_case(st.session_state.user_email, test_case_type, "Details not provided", file.name)
                st.success(f"{len(uploaded_file)} screenshots uploaded and saved!")
            else:
                save_test_case(st.session_state.user_email, test_case_type, "Details not provided", uploaded_file.name)
                st.success(f"Test case with file '{uploaded_file.name}' saved successfully!")
        else:
            st.warning("Please upload a file before submitting!")
else:
    st.warning("Please login to use the Test Case Generator.")
