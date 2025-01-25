### AI-Data-Analysis-Agent

📊 AI Data Analyst Agent is a Streamlit-based web application that allows users to upload data files (CSV or Excel) and interact with them using natural language queries. The app leverages theleverages the Groq API to generate SQL queries based on user questions and executes them using DuckDB. The results are displayed in an interactive and user-friendly interface.

🚀 Features
File Upload: Upload CSV or Excel files.

Data Preview: View a preview of the uploaded data.

Natural Language Querying: Ask questions about the data in plain English.

SQL Query Generation: Automatically generates SQL queries using the Groq API.

Query Execution: Executes the generated SQL queries using DuckDB.

Result Display: Displays query results in a tabular format.

📋 Pre-Installation Requirements
Before running the application, ensure you have the following installed:

Python 3.8 or higher: The application is built using Python.

Streamlit: A framework for building web applications.

DuckDB: An in-process SQL OLAP database management system.

Groq API Key: You need a Groq API key to interact with the Groq API.

🛠️ Installation
Clone the Repository:

bash
Copy
git clone https://github.com/your-username/smart-data-analyst.git
cd smart-data-analyst
Create a Virtual Environment (Optional but recommended):

bash
Copy
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies:

bash
Copy
pip install -r requirements.txt
Set Up Groq API Key:

Obtain your Groq API key from the Groq Cloud.

Create a .env file in the root directory and add your API key:

plaintext
Copy
GROQ_API_KEY=your_api_key_here
🏃‍♂️ Running the Application
Run the Streamlit App:

bash
Copy
streamlit run app.py
Access the App:

Open your web browser and navigate to http://localhost:8501.

Upload a File:

Use the file uploader to upload a CSV or Excel file.

Ask Questions:

Enter your question in the text area and click "Analyze" to generate insights.
