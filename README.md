# 📊 Bank Risk Controller System  

### 🖥️ Developed by: **Mohammed Riyaskhan**  
📧 **Email**: [Riyaziqooz311@gmail.com](mailto:Riyaziqooz311@gmail.com)  
🔗 **LinkedIn**: [linkedin.com/in/riyas-khan-36a748278](https://www.linkedin.com/in/riyas-khan-36a748278)  

---

## 📖 Table of Contents  
1. [📂 Introduction](#introduction)  
2. [❓ Problem Statement](#problem-statement)  
3. [✨ Features of the Application](#features-of-the-application)  
4. [📊 Performance Metrics](#performance-metrics)  
5. [🛠️ Technologies Used](#technologies-used)  
6. [📸 Screenshots](#screenshots)  
7. [🎯 Conclusion](#conclusion)  

---

## 📂 Introduction  
The **Bank Risk Controller System** is a cutting-edge solution leveraging machine learning to predict loan defaults and optimize banking operations. This interactive system includes a Streamlit dashboard for visualization, prediction, and a chatbot powered by Hugging Face embeddings to enhance customer engagement.  

---

## ❓ Problem Statement  
Predict whether a loan customer will default using historical loan data.  

### Objectives:  
- 🛡️ **Risk Management**: Assess and mitigate risks in loan approvals.  
- 👥 **Customer Segmentation**: Tailor financial products based on risk profiles.  
- 📈 **Credit Scoring**: Enhance accuracy using predictive analytics.  
- 🔍 **Fraud Detection**: Identify patterns of fraudulent loan applications.  

The target variable in the dataset is **`TARGET`**, indicating whether the customer defaulted.  

---

## ✨ Features of the Application  

### 🌟 Core Sidebars  
1. **📂 Data**  
   - Display the dataset used for model building.  
   - Show model performance metrics.  

2. **📊 EDA - Visual**  
   - Perform exploratory data analysis (EDA).  
   - Display interactive visualizations using Plotly.  

3. **🤖 Prediction**  
   - Accept user input for selected features.  
   - Predict whether the customer defaulted or not.  

4. **💬 Bank Chatbox**  
   - Interactive chatbot to handle customer queries.  
   - Powered by Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`).  

---

## 📊 Performance Metrics  

| **📈 Model**               | **✔️ Accuracy** | **🎯 Precision** | **🔄 Recall** | **📐 F1 Score** | **🔍 ROC AUC** | **📋 Confusion Matrix**                 |  
|----------------------------|----------------|------------------|---------------|----------------|---------------|----------------------------------------|  
| **ExtraTreesClassifier**   | 1.0000         | 1.0000           | 1.0000        | 1.0000         | 0.9999        | `[[161689, 5], [0, 162520]]`           |  
| **RandomForestClassifier** | 0.9999         | 0.9999           | 0.9999        | 0.9999         | 0.9999        | `[[161675, 19], [0, 162520]]`          |  
| **DecisionTreeClassifier** | 0.9971         | 0.9971           | 0.9971        | 0.9971         | 0.9970        | `[[160746, 948], [0, 162520]]`         |  
| **XGBoostClassifier**      | 0.7661         | 0.7665           | 0.7661        | 0.7660         | 0.7660        | `[[120689, 41005], [34833, 127687]]`   |  

---

## 🛠️ Technologies Used  
- **📜 Programming Language**: Python  
- **📚 Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Plotly, Seaborn, Streamlit  
- **🤖 Machine Learning Models**:  
  - ExtraTreesClassifier  
  - RandomForestClassifier  
  - DecisionTreeClassifier  
  - XGBoostClassifier  
- **💬 Generative AI**: Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`)  

---

## 📸 Screenshots  

1. **📂 Data and Metrics Display**  
   - Showcase of dataset and model performance metrics.  

2. **📊 EDA Visualizations**  
   - Interactive plots and statistics display.  

3. **🤖 Prediction Sidebar**  
   - User input for features and predicted output.  

4. **💬 Bank Chatbot**  
   - Chatbot interface responding to customer queries.  

---

## 🎯 Conclusion  
The **Bank Risk Controller System** is a state-of-the-art application designed for the banking sector to improve decision-making processes. With its robust machine learning models and AI-powered chatbot, the project ensures seamless user experience and impactful results.  

📧 **Email**: [Riyaziqooz311@gmail.com](mailto:Riyaziqooz311@gmail.com)  
🔗 **LinkedIn**: [linkedin.com/in/riyas-khan-36a748278](https://www.linkedin.com/in/riyas-khan-36a748278)  
