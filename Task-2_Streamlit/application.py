import streamlit as st
import prediction

st.markdown("<h1 style='text-align: center;'>Multi-Modal Candidate Shortlisting</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>System</h1>", unsafe_allow_html=True)


st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Home", "About"])


with tab1:

    form_values = {
        "job_category": None,
        "resume_text": None
    }

    with st.form(key="resume_form"):
        st.write("Fill out the form below to find out if you are eligible for interviews at our organisation.")

        form_values['job_category'] = st.selectbox(
            'Choose the job you want to apply for:',
            ['-------', 'Data Science', 'HR', 'Advocate', 'Arts', 'Web Designing',
            'Mechanical Engineer', 'Sales', 'Health and fitness',
            'Civil Engineer', 'Java Developer', 'Business Analyst',
            'SAP Developer', 'Automation Testing', 'Electrical Engineering',
            'Operations Manager', 'Python Developer', 'DevOps Engineer',
            'Network Security Engineer', 'PMO', 'Database', 'Hadoop',
            'ETL Developer', 'DotNet Developer', 'Blockchain', 'Testing']
        )

        form_values['resume_text'] = st.text_input("Enter your resume (select the text in your resume, and paste it below):")

        submit_button = st.form_submit_button()

        if submit_button:
            if not all(form_values.values()):
                st.warning("Please fill in all of the fields.")
            else:
                prediction = prediction.Prediction(form_values['resume_text'], form_values['job_category'])
                if prediction.prediction() == form_values['job_category']:
                    st.write("Congrats! You have been shortlisted for the interview stage.")
                    st.balloons()
                else:
                    st.write("Sorry! You were not shortlisted. Better luck next time!!")


with tab2:
    st.markdown("""
    <p style='text-align: center;'>
    This system uses a deep learning model to extract important features from your resume, and use it to predict the appropriate job category.
    </p>
    <p style='text-align: center;'>
    If the predicted job matches the one you filled in the form, you will be shortlisted for the interview process.
    </p>
    """, unsafe_allow_html=True)
