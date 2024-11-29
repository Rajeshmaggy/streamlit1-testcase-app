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
if st.session_state.logged_in:
    st.markdown(
        f"<div style='position: absolute; top: 20px; right: 20px;'>Logged in as: {st.session_state.user_email}</div>",
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
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 400px;
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.2);
            z-index: 10;">
            <h3 style="text-align: center; font-family: Arial, sans-serif; margin-bottom: 20px;">Login</h3>
            <div style="text-align: center; margin-bottom: 20px;">
                <button style="
                    width: 100%;
                    padding: 10px;
                    background-color: #4285F4;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                ">Sign in with Google</button>
            </div>
            <div style="text-align: center; margin: 10px 0;">
                <span style="color: #aaa;">or</span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Email and Password inputs inside the modal
    email = st.text_input("Email", key="auth_email", placeholder="Your email")
    password = st.text_input("Password", type="password", key="auth_password", placeholder="Your password")
    remember_me = st.checkbox("Remember me", key="remember_me")

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
                st.success("Signup successful! Please login.")
                st.session_state.show_modal = False
                st.experimental_rerun()

    # Close button
    if st.button("Close", key="close_modal"):
        st.session_state.show_modal = False
        st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# Main Test Case Generator content
if st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center;'>Test Case Generator</h1>", unsafe_allow_html=True)
    st.write("Welcome to the Test Case Generator! You are now logged in.")
else:
    st.warning("Please login to access the Test Case Generator.")
