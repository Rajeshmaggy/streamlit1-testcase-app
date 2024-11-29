import streamlit as st
import pandas as pd
import os

# Set up the page
st.set_page_config(page_title="Test Case Generator", layout="wide")

# File path to store data
DATA_FILE = "test_cases.csv"

# Load existing data or create a new DataFrame
if os.path.exists(DATA_FILE):
    test_cases_df = pd.read_csv(DATA_FILE)
else:
    test_cases_df = pd.DataFrame(columns=["Email", "Test Case Type", "Test Case Details", "File Name"])


# Function to save data
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


# Sidebar: User email input
st.sidebar.header("User Info")
user_email = st.sidebar.text_input("Enter your email address:")
if user_email:
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

# Test case details input
test_case_details = st.text_area(
    "Enter Test Case Details:",
    help="Provide details about the test case you are performing.",
)

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
    if not user_email:
        st.warning("Please enter your email address before submitting.")
    elif not test_case_details.strip():
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

# Allow the user to download all test cases in CSV format
# st.markdown("---")
# st.markdown("### Download All Test Cases")
# csv_data = test_cases_df.to_csv(index=False)
# st.download_button(
#     label="Download CSV",
#     data=csv_data,
#     file_name="all_test_cases.csv",
#     mime="text/csv",

