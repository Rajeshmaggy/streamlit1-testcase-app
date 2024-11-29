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
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "show_form" not in st.session_state:
    st.session_state.show_form = False

# Custom CSS for positioning the button
st.markdown(
    """
    <style>
    .top-right-button {
        position: absolute;
        top: 10px;
        right: 20px;
        z-index: 1000;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Top-right button
if not st.session_state.logged_in:
    st.markdown(
        '<button class="top-right-button" onclick="document.getElementById(\'auth-form\').style.display=\'block\'">Login / Signup</button>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<button class="top-right-button" disabled>Welcome, {st.session_state.user_email}</button>',
        unsafe_allow_html=True,
    )

# Authentication form
if not st.session_state.logged_in:
    with st.container():
        if st.session_state.show_form or st.button("Show Login / Signup Form"):
            st.session_state.show_form = True
            st.write("### User Authentication")
            auth_option = st.radio("Choose an option:", ["Login", "Signup"], key="auth_option")
            email = st.text_input("Email", key="auth_email")
            password = st.text_input("Password", type="password", key="auth_password")

            if auth_option == "Login":
                if st.button("Login"):
                    if verify_login(email, password):
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.session_state.show_form = False
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
                        st.session_state.show_form = False
                        st.experimental_rerun()

# Main content
if st.session_state.logged_in:
    st.success(f"Welcome {st.session_state.user_email}! You are logged in.")
    st.text_area("Test Case Generator:", "Enter your test case details here...")
else:
    st.warning("Please login to use the Test Case Generator.")
