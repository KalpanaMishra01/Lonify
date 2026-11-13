Lonify 🏦 – Loan Approval Prediction System

Lonify is a Machine Learning-based Loan Approval Prediction System designed to automate the loan evaluation process and assist in making data-driven lending decisions.

The project uses a Gaussian Naive Bayes classifier to predict whether a loan application should be approved or rejected based on applicant financial and personal information.

Features

- Loan approval prediction using Machine Learning.
- Interactive interface built with Streamlit.
- Applicant financial and personal details input.
- Data preprocessing and feature encoding.
- Automated loan approval/rejection prediction.

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

Machine Learning Model

Algorithm: Gaussian Naive Bayes

The model uses preprocessed applicant information to generate loan approval predictions.

Project Structure

Lonify/
│
├── app.py
├── loan_modell.pkl
├── preprocessing.pkl
├── requirements.txt
├── .gitignore
└── README.md

Installation and Usage

1. Clone the repository:

git clone YOUR_REPOSITORY_URL

2. Navigate to the project directory:

cd Lonify

3. Install the required dependencies:

pip install -r requirements.txt

4. Run the Streamlit application:

streamlit run app.py

5. Open the local URL provided by Streamlit in your browser.

Dataset

The dataset used for training is not included in this repository.

Project Objective

The main objective of Lonify is to reduce manual effort in loan evaluation and support more consistent lending decisions by identifying applicants who are more likely to qualify for a loan.