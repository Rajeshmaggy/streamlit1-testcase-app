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
        /* Style for the Login/Signup button */
        .login-button {
            position: fixed;
            top: 10px;
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

        /* Modal styling */
        .modal {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 400px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.25);
            z-index: 1001;
            padding: 20px;
        }

        /* Overlay to dim the background */
        .overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.4);
            z-index: 1000;
        }

        /* Close button inside modal */
        .close-button {
            background-color: #f44336;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 5px 10px;
            cursor: pointer;
            float: right;
        }
        .close-button:hover {
            background-color: #d32f2f;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Login/Signup button in the top-right corner
st.markdown(
    """
    <button class="login-button" onclick="window.dispatchEvent(new Event('showModal'))">Login / Signup</button>
    """,
    unsafe_allow_html=True,
)

# Handle modal display
if st.button("Login / Signup", key="open_modal"):
    st.session_state.show_modal = True

if st.session_state.show_modal:
    # Add overlay effect
    st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)

    # Modal content
    st.markdown(
        """
        <div class="modal">
            <h3 style="text-align: center;">Login or Signup</h3>
            <hr>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Modal form
    auth_option = st.radio("Choose an option:", ["Login", "Signup"], key="auth_option")
    email = st.text_input("Email", key="auth_email")
    password = st.text_input("Password", type="password", key="auth_password")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Login"):
            if auth_option == "Login" and verify_login(email, password):
                st.success("Login successful!")
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
                st.success("Signup successful! Please login.")
                st.session_state.show_modal = False
                st.experimental_rerun()

    # Close button
    if st.button("Close", key="close_modal"):
        st.session_state.show_modal = False
        st.experimental_rerun()

# Test Case Generator Main Content
st.title("Test Case Generator")
if st.session_state.logged_in:
    st.sidebar.write(f"Logged in as: {st.session_state.user_email}")

    test_case_type = st.selectbox("Select Test Case Type:", ["Video", "Screenshot", "Document"])
    uploaded_file = st.file_uploader("Upload your file:", type=["mp4", "jpg", "pdf"])

    if st.button("Submit"):
        if uploaded_file:
            st.success(f"Test case submitted with file: {uploaded_file.name}")
        else:
            st.warning("Please upload a file before submitting.")
else:
    st.warning("Please login to access the Test Case Generator.")
