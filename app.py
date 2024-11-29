# Add a Sidebar for User History
if st.session_state.login_successful:
    # Filter test cases for the logged-in user
    user_history = test_cases_df[test_cases_df["Email"] == st.session_state.user_email]

    with st.sidebar:
        st.markdown("### User History")
        if user_history.empty:
            st.info("No test cases submitted yet.")
        else:
            # Allow filtering by Test Case Type
            filter_type = st.selectbox(
                "Filter by Test Case Type:",
                ["All"] + user_history["Test Case Type"].unique().tolist()
            )
            
            if filter_type != "All":
                filtered_history = user_history[user_history["Test Case Type"] == filter_type]
            else:
                filtered_history = user_history

            # Display filtered history
            st.dataframe(filtered_history[["Test Case Type", "File Name"]].reset_index(drop=True))

# Main Section - Test Case Generator (Same as Before)
st.markdown("<h1 style='text-align: center;'>Test Case Generator</h1>", unsafe_allow_html=True)

# Profile Icon (No Changes Here)
user_initial = st.session_state.user_email[0].upper()
st.markdown(
    f"""
    <div class="profile-icon" onclick="showProfileInfo()">
        <div class="profile-circle">{user_initial}</div>
    </div>
    <script>
    function showProfileInfo() {{
        var profileDiv = document.getElementById('profile-info');
        if (profileDiv.style.display === "none" || !profileDiv.style.display) {{
            profileDiv.style.display = "block";
        }} else {{
            profileDiv.style.display = "none";
        }}
    }}
    </script>
    """,
    unsafe_allow_html=True
)

# Right-Aligned Profile Info Box (Same as Before)
st.markdown(
    f"""
    <div id="profile-info" style="display: none; position: fixed; top: 65px; right: 20px; width: 300px; background-color: #f9f9f9; border: 1px solid #ddd; padding: 15px; z-index: 1000; text-align: right;">
        <h4 style="margin-bottom: 10px;">Profile Info</h4>
        <p><strong>Email:</strong> {st.session_state.user_email}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Form Inputs for New Test Cases
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    test_case_type = st.selectbox(
        "Test Case Type",
        ["Test Case Generation", "Test Case Validation", "Context Modeling"],
        help="Choose the type of test case you are working on."
    )

with col2:
    content_type = None
    if test_case_type == "Test Case Generation":
        content_type = st.selectbox(
            "Content Type",
            ["Photo", "Video", "Document"],
            help="Choose the type of content related to this test case."
        )

with col3:
    uploaded_file = None
    if content_type == "Photo":
        uploaded_file = st.file_uploader("Upload Photo", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    elif content_type == "Video":
        uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    elif content_type == "Document":
        uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])

# Submit Button
if st.button("Submit Test Case"):
    if 'user_email' not in st.session_state:
        st.warning("Please log in to submit a test case.")
    elif not uploaded_file:
        st.warning("Please upload a file for the selected content type.")
    else:
        # Handle multiple photos
        if isinstance(uploaded_file, list):  # For multiple files (photos)
            for file in uploaded_file:
                save_data(st.session_state.user_email, test_case_type, file.name)
            st.success(f"{len(uploaded_file)} photos uploaded and test case saved successfully!")
        else:  # For single file (video/document)
            save_data(st.session_state.user_email, test_case_type, uploaded_file.name)
            st.success(f"Test case with file '{uploaded_file.name}' saved successfully!")
