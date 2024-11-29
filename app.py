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
if 'show_profile_info' not in st.session_state:
    st.session_state.show_profile_info = False

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

# Sidebar for user history and upload section
def display_sidebar():
    with st.sidebar:
        st.markdown("### Upload and History")

        # Show file upload section
        uploaded_file = None
        if test_case_type == "Test Case Generation":
            uploaded_file = st.file_uploader("Upload Photo", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        elif test_case_type == "Test Case Validation":
            uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
        elif test_case_type == "Context Modeling":
            uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])

        if uploaded_file:
            if isinstance(uploaded_file, list):
                for file in uploaded_file:
                    save_data(st.session_state.user_email, test_case_type, file.name)
                st.success(f"{len(uploaded_file)} files uploaded successfully!")
            else:
                save_data(st.session_state.user_email, test_case_type, uploaded_file.name)
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        # Show user history section (in text format)
        st.markdown("### Your Test Cases")
        user_history = test_cases_df[test_cases_df["Email"] == st.session_state.user_email]
        if user_history.empty:
            st.write("No test cases submitted yet.")
        else:
            for index, row in user_history.iterrows():
                st.write(f"- **Test Case Type**: {row['Test Case Type']} | **File**: {row['File Name']}")

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

    # Profile Icon and Button to toggle profile info near the icon
    user_initial = st.session_state.user_email[0].upper()

    # Create the layout for the profile icon and button in the right corner
    st.markdown("""
        <style>
            .profile-container {
                position: fixed;
                top: 60px;
                right: 10px;
                z-index: 1000;
                display: flex;
                align-items: center;
            }
            .profile-icon {
                border-radius: 50%;
                background-color: #4CAF50;
                color: white;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                font-weight: bold;
                cursor: pointer;
            }
            #profile-info {
                position: fixed;
                top: 60px;
                right: 10px;
                background-color: #f9f9f9;
                border: 1px solid #ddd;
                padding: 15px;
                width: 300px;
                z-index: 1000;
                text-align: right;
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

    # Profile icon and button
    st.markdown(f"""
        <div class="profile-container">
            <div class="profile-icon" onclick="toggleProfileInfo()">
                {user_initial}
            </div>
        </div>
        <script>
            function toggleProfileInfo() {{
                var profileDiv = document.getElementById('profile-info');
                if (profileDiv.style.display === "none" || !profileDiv.style.display) {{
                    profileDiv.style.display = "block";
                }} else {{
                    profileDiv.style.display = "none";
                }}
            }}
        </script>
    """, unsafe_allow_html=True)

    # Display Profile Info if toggled
    if st.session_state.show_profile_info:
        st.markdown(f"""
            <div id="profile-info">
                <h4>Profile Info</h4>
                <p><strong>Email:</strong> {st.session_state.user_email}</p>
            </div>
        """, unsafe_allow_html=True)

    # Test case form and file uploads
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

    display_sidebar()
