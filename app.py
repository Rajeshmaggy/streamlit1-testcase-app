import streamlit as st
import pandas as pd
import os

# Set up the page
st.set_page_config(page_title="Test Case Generator", layout="wide")

# File paths for storing user data
USERS_FILE = "users.csv"

# Load or create user data
if os.path.exists(USERS_FILE):
    users_df = pd.read_csv(USERS_FILE)
else:
    users_df = pd.DataFrame(columns=["Email", "Password"])

# Functions for authentication
def add_user(email, password):
    """Add a new user."""
    global users_df
    new_user = pd.DataFrame({"Email": [email], "Password": [password]})
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv(USERS_FILE, index=False)

def verify_login(email, password):
    """Verify user login credentials."""
    return any((users_df["Email"] == email) & (users_df["Password"] == password))

# Session states
if "show_modal" not in st.session_state:
    st.session_state.show_modal = False
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

# Header with login/signup button
if not st.session_state.logged_in:
    if st.button("Login / Signup", key="login_button"):
        st.session_state.show_modal = True
else:
    st.button(f"Logged in as {st.session_state.user_email}", disabled=True)

# Modal for login/signup
if st.session_state.show_modal:
    # Add a modal effect
    st.markdown(
        """
        <style>
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 1000;
        }
        .modal-content {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 400px;
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            z-index: 1100;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.25);
        }
        </style>
        <div class="modal-overlay"></div>
        <div class="modal-content">
        """,
        unsafe_allow_html=True,
    )

    # Login/Signup modal content
    st.write("### User Authentication")
    auth_option = st.radio("Choose an option:", ["Login", "Signup"], key="auth_option")
    email = st.text_input("Email", key="auth_email")
    password = st.text_input("Password", type="password", key="auth_password")

    if auth_option == "Login":
        if st.button("Login"):
            if verify_login(email, password):
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.show_modal = False
                st.experimental_rerun()
            else:
                st.error("Invalid email or password.")
    elif auth_option == "Signup":
        if st.button("Signup"):
            if email in users_df["Email"].values:
                st.error("This email is already registered.")
            else:
                add_user(email, password)
                st.success("Signup successful! Please login.")
                st.session_state.show_modal = False
                st.experimental_rerun()

    # Close button
    if st.button("Close"):
        st.session_state.show_modal = False

    st.markdown("</div>", unsafe_allow_html=True)

# Main content (background)
if not st.session_state.logged_in:
    st.warning("Please login to use the Test Case Generator.")
else:
    st.success(f"Welcome {st.session_state.user_email}! You are logged in.")
    st.text_area("Test Case Generator:", "Enter your test case details here...")
