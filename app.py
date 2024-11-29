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

# Functions
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

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "show_modal" not in st.session_state:
    st.session_state.show_modal = False

# CSS for styling the modal and button
st.markdown(
    """
    <style>
        /* Style for the top-right Login/Signup button */
        .login-button {
            position: fixed;
            top: 60px;
            right: 20px;
            background-color: #007BFF;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 15px;
            font-size: 14px;
            cursor: pointer;
            z-index: 1000;
        }
        .login-button:hover {
            background-color: #0056b3;
        }

        /* Style for the modal popup */
        .modal {
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
            z-index: 1001;
        }

        /* Background dimming effect when modal is open */
        .overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.3);
            z-index: 1000;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top-right corner Login/Signup button
st.markdown(
    f"""
    <button class="login-button" onclick="window.dispatchEvent(new Event('open_modal'))">
        Login / Signup
    </button>
    """,
    unsafe_allow_html=True,
)

# Handle modal display
if st.button("Open Login Modal"):
    st.session_state.show_modal = True

# Show the modal if triggered
if st.session_state.show_modal:
    # Overlay for dimmed background
    st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)

    # Modal content
    st.markdown(
        """
        <div class="modal">
            <h3 style="text-align: center;">User Authentication</h3>
            <hr>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Authentication form inside modal
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        auth_option = st.radio("Choose an option:", ["Login", "Signup"], key="auth_option")
        email = st.text_input("Email", key="auth_email")
        password = st.text_input("Password", type="password", key="auth_password")

        if auth_option == "Login" and st.button("Login", key="login_button"):
            if verify_login(email, password):
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.show_modal = False
                st.experimental_rerun()
            else:
                st.error("Invalid email or password.")
        elif auth_option == "Signup" and st.button("Signup", key="signup_button"):
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

# Main Test Case Generator content
if st.session_state.logged_in:
    st.sidebar.header("User Info")
    st.sidebar.write(f"Logged in as: **{st.session_state.user_email}**")

    st.markdown("<h1 style='text-align: center;'>Test Case Generator</h1>", unsafe_allow_html=True)

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
            st.success(f"Test case with file '{uploaded_file.name}' saved successfully!")
        else:
            st.warning("Please upload a file before submitting!")
else:
    st.warning("Please login to use the Test Case Generator.")
