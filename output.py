import pandas as pd
import numpy as np
import pickle
import streamlit as st
import os
import re
import itertools
import fitz 

#model_imports
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns

#chat_bot_imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import pdfplumber
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px 


# Add CSS for enhanced visuals!
st.markdown("""
    <style>
        /* General Page Styling */
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f9f9f9;
        }
        
        /* Sidebar Styling */
        .css-1d391kg {
            background-color: #ffffff !important;
            border-right: 2px solid #007BFF;
        }

        /* Sidebar Options Styling */
        div[data-testid="stSidebar"] .css-1lcbmhc {
            padding: 10px;
            border-radius: 8px;
        }
        div[data-testid="stSidebar"] .css-1lcbmhc:hover {
            background-color: #e6f2ff;
            box-shadow: 0px 4px 6px rgba(0, 123, 255, 0.2);
        }

        /* Header Styling */
        h1, h2, h3 {
            color: #2e5984;
            font-weight: bold;
        }

        /* Button Styling */
        div.stButton > button {
            background-color: #007BFF;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }
        div.stButton > button:hover {
            background-color: #0056b3;
            transform: scale(1.05);
        }

        /* Dataframe Styling */
        .dataframe {
            border: 1px solid #dee2e6;
            border-radius: 8px;
            background-color: #ffffff;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        /* File Uploader Styling */
        div[data-testid="stFileUploadDropzone"] {
            border: 2px dashed #007BFF !important;
            border-radius: 8px;
            background-color: #f7faff !important;
            padding: 10px;
        }

        /* Plot Titles */
        .plotly-title {
            font-size: 18px;
            font-weight: bold;
            color: #333333;
        }

        /* Cards for Metrics */
        .metric-card {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
            text-align: center;
        }
        .metric-card h3 {
            color: #007BFF;
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)


# Layout with Columns for Visual Separation
st.markdown("<h1 style='color: yellow;'>Bank Loan Clients Dashboard</h1>", unsafe_allow_html=True)


# Sidebar menu with option names and images
menu_images = {
    "Data Display": "c:/Users/USER/Downloads/png-clipart-business-process-information-data-analysis-management-business-people-management-consulting-thumbnail-removebg-preview.png",
    "Visualization": "c:/Users/USER/Downloads/png-clipart-business-process-information-data-analysis-management-business-people-management-consulting-thumbnail-removebg-preview.png",
    "Prediction": "c:/Users/USER/Downloads/png-clipart-business-process-information-data-analysis-management-business-people-management-consulting-thumbnail-removebg-preview.png",
    "Bank Chatbot": "c:/Users/USER/Downloads/png-clipart-business-process-information-data-analysis-management-business-people-management-consulting-thumbnail-removebg-preview.png",
 }

# Sidebar menu
with st.sidebar:
     # Display the image at the top of the sidebar
    st.image(menu_images["Data Display"], use_column_width=True)  
    selected_menu = option_menu(
        menu_title="Navigation",
        options=["Data Display", "Visualization", "Prediction", "Bank Chatbot"],
        styles={
            "nav-link": {"font-size": "18px", "font-family": "Segoe UI"},
            "nav-link-selected": {"background-color": "#7B68EE", "color": "white"}
        }
    )

 
#load dataset
with open('c:/Users/USER/Downloads/ET_Classifier_model.pkl', 'rb') as file:
    model=pickle.load(file)

with open('c:/Users/USER/Downloads/label_encoders.pkl', 'rb') as en:
    model=pickle.load(en)

# Function to validate the model
def validate_model(model):
    if not hasattr(model, 'predict'):
        st.error("Invalid model. Please load the correct machine learning model.")
        return False
    return True


if selected_menu =="Data Display":
    st.header("📊 Data Display")
    df=pd.read_csv("c:/Users/USER/Downloads/model_data.xls")
    metrics = {
    'Model': ['ExtraTreesClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier', 'XGBoostClassifier'],
    'Accuracy': [f"{round(1.0000, 4):.4f}", f"{round(0.9999, 4):.4f}", f"{round(0.9971, 4):.4f}", f"{round(0.7661, 4):.4f}"],  
    'Precision': [f"{round(1.0000, 4):.4f}", f"{round(0.9999, 4):.4f}", f"{round(0.9971, 4):.4f}", f"{round(0.7665, 4):.4f}"],
    'Recall': [f"{round(1.0000, 4):.4f}", f"{round(0.9999, 4):.4f}", f"{round(0.9971, 4):.4f}", f"{round(0.7661, 4):.4f}"],
    'F1 Score': [f"{round(1.0000, 4):.4f}", f"{round(0.9999, 4):.4f}", f"{round(0.9971, 4):.4f}", f"{round(0.7660, 4):.4f}"],
    'ROC AUC': [f"{round(0.9999, 4):.4f}", f"{round(0.9999, 4):.4f}", f"{round(0.9970, 4):.4f}", f"{round(0.7660, 4):.4f}"],
    'Confusion Matrix': [
        '[[161689, 5], [0, 162520]]',
        '[[161675, 19], [0, 162520]]',
        '[[160746, 948], [0, 162520]]',
        '[[120689, 41005], [34833, 127687]]'
    ]
}
 
    metrics_df=pd.DataFrame(metrics)
    st.dataframe(metrics_df)
    st.dataframe(df.head(10))


elif selected_menu == "Visualization":
    sample = pd.read_csv("c:/Users/USER/Downloads/eda_data.xls")
    st.header("📈 Data Visualization")
    
    # Gender Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(":violet[Distribution of Gender in Dataset]")
        pie_colors = ["#ff006e", "#ffd60a", '#6a0dad', '#ff4500']
        sample["CODE_GENDER"].value_counts().plot.pie(
            autopct="%1.1f%%", colors=pie_colors, startangle=90, wedgeprops={"edgecolor": "k"}
        )
        plt.title("Gender Distribution")
        plt.ylabel("")  # Remove default ylabel
        st.pyplot(plt.gcf())
        plt.clf()
        st.markdown(
            '''
            <p style="color: #ffee32; font-size: 30px; text-align: center;">
                <strong><span style="color: #ef233c; font-size: 40px ">67%</span> Female and 
                <span style="color: #ef233c; font-size: 40px ">33%</span> Male</strong>
            </p>''',
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader(":violet[Distribution of Target]")
        pie_colors = ['#d00000', '#ffd500', '#f72585', '#ff4500']
        sample["TARGET"].value_counts().plot.pie(
            autopct="%1.1f%%", colors=pie_colors, startangle=90, wedgeprops={"edgecolor": "k"}
        )
        plt.title("Target Distribution")
        plt.ylabel("")
        st.pyplot(plt.gcf())
        plt.clf()
        st.markdown(
            '''
            <p style="color: #ffee32; font-size: 20px; text-align: center;">
                <strong>1 - Defaulter, 0 - Non-defaulter<br>
                He/she had late payment more than X days are defaulters.<br>
                <span style="color: #ef233c; font-size: 40px ">8%</span> Defaulters, 
                <span style="color: #ef233c; font-size: 40px ">92%</span> Non-defaulters</strong>
            </p>''',
            unsafe_allow_html=True,
        )

    # Loan Type Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(":violet[Distribution of Contract Types in Loan Data]")
        pie_colors = ['#ff006e', '#d00000', '#f72585', '#ff4500']
        sample["NAME_CONTRACT_TYPE_x"].value_counts().plot.pie(
            autopct="%1.1f%%", colors=pie_colors, startangle=90, wedgeprops={"edgecolor": "k"}
        )
        plt.title("Loan Type Distribution")
        plt.ylabel("")
        st.pyplot(plt.gcf())
        plt.clf()
        st.markdown(
            '''
            <p style="color: #ffee32; font-size: 20px; text-align: center;">
                <strong><span style="color: #ef233c; font-size: 40px ">8%</span> Revolving loans<br>
                are a form of credit that allows borrowers to withdraw, repay, and withdraw again up to a certain limit.</strong>
            </p>''',
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader(":violet[Distribution of Loan Type by Gender]")
        sns.countplot(
            x="NAME_CONTRACT_TYPE_x", hue="CODE_GENDER", data=sample,
            palette=["#00bbf9", "#f15bb5", "#ee964b"]
        )
        plt.title("Loan Type by Gender")
        plt.ylabel("Count")
        st.pyplot(plt.gcf())
        plt.clf()


    
# Visualization 1: Distribution of Income and Credit Amounts (col1 and col2)
    col1, col2 = st.columns(2)
    with col1:
     st.subheader(":violet[Distribution of Income by TARGET]")
    fig_income = px.histogram(
        sample, 
        x="AMT_INCOME_TOTAL", 
        color="TARGET", 
        nbins=50, 
        marginal="box", 
        title="Distribution of Income (by TARGET)"
    )
    fig_income.update_layout(
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        font=dict(color="#ffee32"),
        title_font=dict(size=16, color="#ffee32")
    )
    st.plotly_chart(fig_income, use_container_width=True)


    col1, col2 = st.columns(2)
    with col1:
     st.subheader(":violet[Distribution of Credit Amount by TARGET]")
    fig_credit = px.histogram(
        sample, 
        x="AMT_CREDIT_x", 
        color="TARGET", 
        nbins=50, 
        marginal="box", 
        title="Distribution of Credit Amount (by TARGET)"
    )
    fig_credit.update_layout(
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        font=dict(color="#ffee32"),
        title_font=dict(size=16, color="#ffee32")
    )
    st.plotly_chart(fig_credit, use_container_width=True)



# Visualization 2: Age vs Credit Amount Scatter Plot (col1)
    col1, col2 = st.columns(2)
    with col1:
      st.subheader(":violet[Scatter Plot: Age vs Credit Amount]")
    sample['AGE'] = (sample['DAYS_BIRTH'] / -365).astype(int)
    fig_age_credit = px.scatter(
        sample, 
        x="AGE", 
        y="AMT_CREDIT_x", 
        color="TARGET", 
        title="Age vs Credit Amount by TARGET",
        labels={"AGE": "Age (Years)", "AMT_CREDIT_x": "Credit Amount"},
        color_discrete_map={0: "#1f77b4", 1: "#d62728"}
    )
    fig_age_credit.update_layout(
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        font=dict(color="#ffee32"),
        title_font=dict(size=16, color="#ffee32")
    )
    st.plotly_chart(fig_age_credit, use_container_width=True)



# Visualization 3: Loan Annuity by Income Group (col2)
    col1, col2 = st.columns(2)
    with col1:
     st.subheader(":violet[Box Plot: Loan Annuity by Income Group]")
    sample['INCOME_GROUP'] = pd.qcut(sample['AMT_INCOME_TOTAL'], q=4, labels=["Low", "Medium", "High", "Very High"])
    fig_annuity_income = px.box(
        sample, 
        x="INCOME_GROUP", 
        y="AMT_ANNUITY_x", 
        color="TARGET", 
        title="Loan Annuity by Income Group and TARGET",
        labels={"AMT_ANNUITY_x": "Loan Annuity", "INCOME_GROUP": "Income Group"},
        color_discrete_map={0: "#1f77b4", 1: "#d62728"}
    )
    fig_annuity_income.update_layout(
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        font=dict(color="#ffee32"),
        title_font=dict(size=16, color="#ffee32")
    )
    st.plotly_chart(fig_annuity_income, use_container_width=True)


# Visualization 5: Payment Types Breakdown (col2)
    col1, col2 = st.columns(2)
    with col1:
      st.subheader(":violet[Payment Type Breakdown by TARGET]")
    fig_payment = px.histogram(
        sample, 
        x="NAME_PAYMENT_TYPE", 
        color="TARGET", 
        barmode="group", 
        title="Payment Type Breakdown by TARGET",
        text_auto=True
    )
    fig_payment.update_layout(
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        font=dict(color="#ffee32"),
        title_font=dict(size=16, color="#ffee32")
    )
    st.plotly_chart(fig_payment, use_container_width=True)


    col1, col2 = st.columns(2)
    with col1:
     st.subheader(":violet[Loan Amount by Gender]")
    fig_gender_loan = px.box(
        sample, 
        x="CODE_GENDER", 
        y="AMT_CREDIT_x", 
        color="TARGET", 
        title="Loan Amount Distribution by Gender and TARGET",
        labels={"AMT_CREDIT_x": "Loan Amount", "CODE_GENDER": "Gender"},
        color_discrete_map={0: "#1f77b4", 1: "#d62728"}
    )
    fig_gender_loan.update_layout(
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        font=dict(color="#ffee32"),
        title_font=dict(size=16, color="#ffee32")
    )
    st.plotly_chart(fig_gender_loan, use_container_width=True)


    col1, col2 = st.columns(2)
    with col1:
     st.subheader(":violet[Age Distribution by TARGET]")
    fig_age_dist = px.histogram(
        sample, 
        x="AGE", 
        color="TARGET", 
        nbins=40, 
        marginal="box", 
        title="Age Distribution by TARGET",
        labels={"AGE": "Age (Years)"},
        color_discrete_map={0: "#1f77b4", 1: "#d62728"}
    )
    fig_age_dist.update_layout(
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        font=dict(color="#ffee32"),
        title_font=dict(size=16, color="#ffee32")
    )
    st.plotly_chart(fig_age_dist, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
     st.subheader(":violet[Employment Length vs Income Total]")
    fig_employment_income = px.scatter(
        sample, 
        x="DAYS_EMPLOYED", 
        y="AMT_INCOME_TOTAL", 
        color="TARGET", 
        title="Employment Length vs Income by TARGET",
        labels={"DAYS_EMPLOYED": "Employment Length (Days)", "AMT_INCOME_TOTAL": "Income Total"},
        color_discrete_map={0: "#1f77b4", 1: "#d62728"}
    )
    fig_employment_income.update_layout(
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        font=dict(color="#ffee32"),
        title_font=dict(size=16, color="#ffee32")
    )
    st.plotly_chart(fig_employment_income, use_container_width=True)



    col1, col2 = st.columns(2)
    with col1:
     st.subheader(":violet[Default Rate by Education Type]")
    fig_education_default = px.histogram(
        sample, 
        x="NAME_EDUCATION_TYPE", 
        color="TARGET", 
        barmode="group", 
        title="Default Rate by Education Type",
        labels={"NAME_EDUCATION_TYPE": "Education Type"},
        text_auto=True
    )
    fig_education_default.update_layout(
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        font=dict(color="#ffee32"),
        title_font=dict(size=16, color="#ffee32")
    )
    st.plotly_chart(fig_education_default, use_container_width=True)


    col1, col2 = st.columns(2)
    with col1:
     st.subheader(":violet[Income vs Credit Amount by Target]")
    fig_income_credit = px.scatter(
        sample, 
        x="AMT_INCOME_TOTAL", 
        y="AMT_CREDIT_x", 
        color="TARGET", 
        title="Income vs Credit Amount",
        labels={"AMT_INCOME_TOTAL": "Income Total", "AMT_CREDIT_x": "Credit Amount"},
        color_discrete_map={0: "#1f77b4", 1: "#d62728"}
    )
    fig_income_credit.update_layout(
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        font=dict(color="#ffee32"),
        title_font=dict(size=16, color="#ffee32")
    )
    st.plotly_chart(fig_income_credit, use_container_width=True)



    col1, col2 = st.columns(2)
    with col1:
     st.subheader(":violet[Family Size Impact on Default]")
    fig_family_default = px.box(
        sample, 
        x="CNT_FAM_MEMBERS", 
        y="TARGET", 
        color="TARGET", 
        title="Family Size vs Default Rate",
        labels={"CNT_FAM_MEMBERS": "Family Size"},
        color_discrete_map={0: "#1f77b4", 1: "#d62728"}
    )
    fig_family_default.update_layout(
        plot_bgcolor="#1e1e1e", 
        paper_bgcolor="#1e1e1e", 
        font=dict(color="#ffee32"),
        title_font=dict(size=16, color="#ffee32")
    )
    st.plotly_chart(fig_family_default, use_container_width=True)


elif selected_menu == "Prediction":  
    st.header("🔮 Predict Loan Risks")
    st.subheader("Enter the Features for Prediction")

    # Load the dataset
    try:
        model_data = pd.read_csv("c:/Users/USER/Downloads/model_data.xls")
    except FileNotFoundError:
        st.error("The file 'model_data.xls' is not found. Please ensure it is in the specified directory.")
        st.stop()

    # Load encoders
    try:
        encoders = pd.read_pickle('c:/Users/USER/Downloads/label_encoders.pkl')
    except FileNotFoundError:
        st.error("The file 'label_encoders.pkl' is not found. Please ensure it is in the specified directory.")
        st.stop()

    # Retrieve encoder classes
    try:
        ORGANIZATION_TYPE = encoders['ORGANIZATION_TYPE'].classes_.tolist()
        OCCUPATION_TYPE = encoders['OCCUPATION_TYPE'].classes_.tolist()
    except KeyError as e:
        st.error(f"Error retrieving encoder classes: {e}")
        st.stop()

    def get_user_input():
        """Get user input from Streamlit form."""
        st.subheader(":violet[Fill all fields below to check loan status:]")
        cc1, cc2 = st.columns([2, 2])

        with cc1:
            BIRTH_YEAR = st.number_input("Birth Year (YYYY):", min_value=1950, max_value=2024)
            AMT_CREDIT = st.number_input("Credit Amount of Loan:")
            AMT_ANNUITY = st.number_input("Loan Annuity:")
            AMT_INCOME_TOTAL = st.number_input("User's Total Income:")
            ORGANIZATION_TYPE_input = st.selectbox("Organization Type:", ORGANIZATION_TYPE)

        with cc2:
            OCCUPATION_TYPE_input = st.selectbox("Occupation Type:", OCCUPATION_TYPE)
            EXT_SOURCE_2 = st.number_input("External Data Source 2 Score:")
            EXT_SOURCE_3 = st.number_input("External Data Source 3 Score:")
            REGION_POPULATION_RELATIVE = st.number_input("Region Population Relative:")
            HOUR_APPR_PROCESS_START = st.number_input("Hour User Applied for Loan:")
            EMPLOYMENT_START_YEAR = st.number_input("Employment Start Year:", min_value=1950, max_value=2024)

        # User Input Data
        user_input_data = {
            'BIRTH_YEAR': BIRTH_YEAR,
            'AMT_CREDIT': AMT_CREDIT,
            'AMT_ANNUITY': AMT_ANNUITY,
            'AMT_INCOME_TOTAL': AMT_INCOME_TOTAL,
            'ORGANIZATION_TYPE': ORGANIZATION_TYPE_input,
            'OCCUPATION_TYPE': OCCUPATION_TYPE_input,
            'EXT_SOURCE_2': EXT_SOURCE_2,
            'EXT_SOURCE_3': EXT_SOURCE_3,
            'REGION_POPULATION_RELATIVE': REGION_POPULATION_RELATIVE,
            'HOUR_APPR_PROCESS_START': HOUR_APPR_PROCESS_START,
            'EMPLOYMENT_START_YEAR': EMPLOYMENT_START_YEAR
        }

        return pd.DataFrame(user_input_data, index=[0])

    def load_model():
        """Load the Extra Trees Classifier model."""
        try:
            with open('c:/Users/USER/Downloads/ET_Classifier_model.pkl', 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            st.error("Model file not found: ET_Classifier_model.pkl")
            return None
        except Exception as e:
            st.error(f"Error loading model file: {e}")
            return None

    def data_transformation_for_the_model(df):
        """Transform data using pre-loaded encoders."""
        df = df.copy()  # Avoid modifying the original DataFrame
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])
        return df

    # Get user input
    user_data = get_user_input()

    # Load the model
    model = load_model()
    if model is None:
        st.stop()

    # Transform data for prediction
    try:
        transformed_data = data_transformation_for_the_model(user_data)
    except Exception as e:
        st.error(f"Error during data transformation: {e}")
        st.stop()

    # Predict loan risk
    if st.button("Predict Loan Status"):
        try:
            prediction = model.predict(transformed_data)
            st.success("Prediction Successful!")
            result = "Non-Defaulter" if prediction[0] == 0 else "Defaulter"
            st.subheader(f"The user is predicted to be a: {result}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")


# Bank Chatbot Section
elif selected_menu == "Bank Chatbot":
    # PDF Text Extraction Function
    def extract_text_from_pdf(uploaded_file):
        try:
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
                full_text = []
                for page in pdf:
                    full_text.append(page.get_text())
                return "".join(full_text)
        except Exception as e:
            st.error(f"❌ Error extracting text from PDF: {e}")
            return ""

    # Streamlit Interface
    st.title("🏦 Bank Chatbot")
    st.write("🤖 Ask questions related to banking, and I’ll assist you!")

    # File Upload
    uploaded_file = st.file_uploader("📂 Upload a Banking-related PDF", type="pdf")
    show_debug = st.checkbox("🛠️ Show Debug Outputs", value=False)

    # Check session state for previously answered questions
    if "answered_questions" not in st.session_state:
        st.session_state.answered_questions = []

    # Process the Uploaded File
    if uploaded_file is not None:
        # Extract text from the PDF
        with st.spinner("🔍 Extracting text from the PDF..."):
            full_text = extract_text_from_pdf(uploaded_file)
            if full_text.strip() == "":
                st.warning("⚠️ No text extracted from the uploaded PDF.")
                st.stop()
            st.text_area("📜 Extracted PDF Content (Snippet)", full_text[:1000], height=300)

        # Split the text into chunks
        with st.spinner("🧩 Splitting text into chunks..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(full_text)
            documents = [Document(page_content=chunk) for chunk in chunks]

        if show_debug:
            st.write("🛠️ Debug: Sample chunks:", [doc.page_content[:100] for doc in documents[:3]])

        # Create Embeddings and Vector Store
        try:
            with st.spinner("⚙️ Creating vector database..."):
                model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Faster model
                embeddings = HuggingFaceEmbeddings(model_name=model_name)
                vector_db = FAISS.from_documents(documents, embeddings)
        except Exception as e:
            st.error(f"❌ Error creating embeddings or vector database: {e}")
            vector_db = None

        # Load LLM (using a local file or a cloud model)
        model_file = r"c:/Users/USER/Downloads/llama-2-7b-chat.ggmlv3.q8_0.bin"
        llm = None
        if os.path.exists(model_file):
            with st.spinner("🚀 Loading the Llama model..."):
                llm = CTransformers(
                    model=model_file,
                    model_type="llama",
                    config={"max_new_tokens": 200, "temperature": 0.7}
                )
        else:
            st.error("❌ Model file not found.")

        # Initialize RetrievalQA
        if llm and vector_db:
            chatbox_model = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
            )

            # User Question Input
            user_question = st.text_input("💬 Ask a banking question:")

            if user_question.lower() in ["hi", "hello"]:
                st.success("👋 Hello! I'm a bank chatbot, here to assist with your banking queries.")
            elif user_question:
                if user_question.lower() in st.session_state.answered_questions:
                    st.write("❌ You asked this question earlier, no need to repeat.")
                else:
                    if st.button("✨ Get Answer"):
                        with st.spinner("🤔 Fetching the answer..."):
                            try:
                                response = chatbox_model.invoke({"query": user_question})
                                answer = response.get("result", "").strip()
                                if not answer:
                                    answer = "😔 I couldn't find relevant information in the document."
                                st.success(f"✅ Answer: {answer}")
                                # Store the answered question to prevent re-answering
                                st.session_state.answered_questions.append(user_question.lower())
                            except Exception as e:
                                st.error(f"❌ Error: {e}")
        else:
            st.warning("⚠️ Could not initialize the chatbot. Check embeddings and model setup.")
    else:
        st.warning("📥 Please upload a PDF to get started.")
