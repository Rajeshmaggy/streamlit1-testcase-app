import streamlit as st
import pandas as pd
import os

# Set up the page
st.set_page_config(page_title="Test Case Generator", layout="wide")

# File paths for storing data
DATA_FILE = "test_cases.csv"
USERS_FILE = "users.csv"

# Load or create test case data
if os.path.exists(DATA_FILE):
    test_cases_df = pd.read_csv(DATA_FILE)
else:
    test_cases_df = pd.DataFrame(columns=["Email", "Test Case Type", "Test Case Details", "File Name"])

# Load or create user data
if os.path.exists(USERS_FILE):
    users_df = pd.read_csv(USERS_FILE)
else:
    users_df = pd.DataFrame(columns=["Email", "Password"])

# Helper functions
def save_test_case(email, test_case_type, test_case_details, file_name):
    """Save test case details."""
    new_entry = pd.DataFrame({
        "Email": [email],
        "Test Case Type": [test_case_type],
        "Test Case Details": [test_case_details],
        "File Name": [file_name],
    })
    global test_cases_df
    test_cases_df = pd.concat([test_cases_df, new_entry], ignore_index=True)
    test_cases_df.to_csv(DATA_FILE, index=False)

def add_user(email, password):
    """Add a new user."""
    new_user = pd.DataFrame({"Email": [email], "Password": [password]})
    global users_df
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv(USERS_FILE, index=False)

def verify_login(email, password):
    """Verify user login credentials."""
    return any((users_df["Email"] == email) & (users_df["Password"] == password))

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "show_modal" not in st.session_state:
    st.session_state.show_modal = False

# Main Test Case Generator content
st.markdown("<h1 style='text-align: center;'>Test Case Generator</h1>", unsafe_allow_html=True)

if st.session_state.logged_in:
    st.sidebar.header("User Info")
    st.sidebar.write(f"Logged in as: **{st.session_state.user_email}**")

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
    st.warning("Please log in to access the Test Case Generator.")

# Top-right Login/Signup button
button_html = """
<div style="
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;">
    <button onclick="window.dispatchEvent(new Event('modal_open'))"
            style="
                background-color: #007BFF;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px 15px;
                cursor: pointer;
                font-size: 14px;">
        Login / Signup
    </button>
</div>
"""
st.markdown(button_html, unsafe_allow_html=True)

# Display modal popup for login/signup
if st.session_state.show_modal:
    st.markdown(
        """
        <div style="
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 400px;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.2);
            z-index: 10000;">
            <h3 style="text-align: center;">Login / Signup</h3>
            <div style="margin: 20px 0;">
        """,
        unsafe_allow_html=True,
    )

    # Email and password inputs
    email = st.text_input("Email", key="auth_email", placeholder="Enter your email")
    password = st.text_input("Password", type="password", key="auth_password", placeholder="Enter your password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            if verify_login(email, password):
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.show_modal = False
                st.experimental_rerun()
            else:
                st.error("Invalid email or password.")
    with col2:
        if st.button("Signup"):
            if email in users_df["Email"].values:
                st.error("This email is already registered.")
            else:
                add_user(email, password)
                st.success("Signup successful! Please log in.")
                st.session_state.show_modal = False
                st.experimental_rerun()

    if st.button("Close"):
        st.session_state.show_modal = False
        st.experimental_rerun()

    # Close modal tag
    st.markdown("</div>", unsafe_allow_html=True)

# JavaScript to trigger modal popup
st.markdown(
    """
    <script>
    document.addEventListener('modal_open', function () {
        window.parent.postMessage({type: 'SHOW_MODAL'}, '*');
    });
    </script>
    """,
    unsafe_allow_html=True,
)
