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
st.markdown(
    """
    <style>
        .top-right-button {
            position: fixed;
            top: 20px;
            right: 20px;
            background-color: #007BFF;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 10px 15px;
            font-size: 14px;
            cursor: pointer;
            z-index: 100;
        }
        .top-right-button:hover {
            background-color: #0056b3;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.logged_in:
    st.markdown(
        f"<button class='top-right-button' onclick='toggleModal()'>Login / Signup</button>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"<button class='top-right-button'>Logged in as {st.session_state.user_email}</button>",
        unsafe_allow_html=True,
    )

# Modal for login/signup
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
            z-index: 1000;">
            <h3 style="text-align: center; margin-bottom: 20px;">User Authentication</h3>
        """,
        unsafe_allow_html=True,
    )

    # Content inside the modal
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        auth_option = st.radio("Choose an option:", ["Login", "Signup"], key="auth_option")
        email = st.text_input("Email", key="auth_email")
        password = st.text_input("Password", type="password", key="auth_password")

        if auth_option == "Login" and st.button("Login"):
            if verify_login(email, password):
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.show_modal = False
                st.experimental_rerun()
            else:
                st.error("Invalid email or password.")
        elif auth_option == "Signup" and st.button("Signup"):
            if email in users_df["Email"].values:
                st.error("This email is already registered.")
            else:
                add_user(email, password)
                st.success("Signup successful! Please login.")
                st.session_state.show_modal = False
                st.experimental_rerun()

        if st.button("Close"):
            st.session_state.show_modal = False
            st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# Main content (background)
if not st.session_state.logged_in:
    st.warning("Please login to use the Test Case Generator.")
else:
    st.success(f"Welcome {st.session_state.user_email}! You are logged in.")
    st.text_area("Test Case Generator:", "Enter your test case details here...")

# JavaScript to toggle modal visibility
st.markdown(
    """
    <script>
        function toggleModal() {
            const modalVisible = %s;
            if (modalVisible) {
                document.querySelector('.stApp').style.filter = 'blur(0px)';
            } else {
                document.querySelector('.stApp').style.filter = 'blur(4px)';
            }
        }
    </script>
    """ % str(st.session_state.show_modal).lower(),
    unsafe_allow_html=True,
)
