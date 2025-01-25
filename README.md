### AI Data Analyst 

📊 **AI Data Analyst** is a Streamlit-based web application that allows users to upload data files (CSV or Excel) and interact with them using natural language queries. The app leverages the **Groq API** to generate SQL queries based on user questions and executes them using **DuckDB**. The results are displayed in an interactive and user-friendly interface, making data analysis accessible to everyone.

---

## 🚀 Features

- **File Upload**: Upload CSV or Excel files.
- **Data Preview**: View a preview of the uploaded data.
- **Natural Language Querying**: Ask questions about the data in plain English.
- **AI-Powered SQL Query Generation**: Automatically generates SQL queries using the Groq API.
- **Query Execution**: Executes the generated SQL queries using DuckDB.
- **Result Display**: Displays query results in a tabular format.

---

## 📋 Pre-Installation Requirements

Before running the application, ensure you have the following installed:

1. **Python 3.8 or higher**: The application is built using Python.
2. **Streamlit**: A framework for building web applications.
3. **DuckDB**: An in-process SQL OLAP database management system.
4. **Groq API Key**: You need a Groq API key to interact with the Groq API.

---

## 🛠️ Installation


1. **Create a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Groq API Key**:
   - Obtain your Groq API key from the [Groq Cloud](https://groq.com/).
   - Create a `.env` file in the root directory and add your API key:
     ```plaintext
     GROQ_API_KEY=your_api_key_here
     ```

---

## 🏃‍♂️ Running the Application

1. **Run the Streamlit App**:
   ```bash
   streamlit run ai_data_analyst.py
   ```

2. **Access the App**:
   - Open your web browser and navigate to `http://localhost:8501`.

3. **Upload a File**:
   - Use the file uploader to upload a CSV or Excel file.

4. **Ask Questions**:
   - Enter your question in the text area and click "Analyze" to generate insights.

---

## 📂 File Structure

```
ai-data-analyst/
│
├── app.py                # Main Streamlit application code
├── requirements.txt      # List of Python dependencies
├── README.md             # This file
└── .env                  # Environment variables (e.g., Groq API key)
```

---

## 🧑‍💻 Usage Instructions

1. **Upload a Data File**:
   - Click on the "Upload data file" button and select a CSV or Excel file.

2. **Preview the Data**:
   - Once the file is uploaded, a preview of the data will be displayed.

3. **Ask a Question**:
   - Enter a question about the data in the text area provided.
   - Example: "Show distribution of companies by country."

4. **Generate Insights**:
   - Click the "Analyze" button to generate and execute the SQL query.
   - The results will be displayed in a tabular format.

---

## 🛑 Troubleshooting

- **File Upload Issues**:
  - Ensure the file is in the correct format (CSV or Excel).
  - Check for any encoding issues, especially with CSV files.

- **Groq API Errors**:
  - Ensure your Groq API key is correctly set in the `.env` file.
  - Check your internet connection.

- **SQL Query Errors**:
  - Ensure the question is clear and relevant to the data.
  - The app validates the SQL query before execution, so any invalid queries will be flagged.

---

## 🙏 Acknowledgments

- **Streamlit**: For providing an excellent framework for building data apps.
- **Groq**: For their powerful API that enables natural language processing.
- **DuckDB**: For their efficient in-memory SQL database.

---

## 📧 Contact

For any questions or feedback, feel free to reach out:

- **Email**: avi.hm24@gmail.com

---

Enjoy analyzing your data with **AI Data Analyst**! 🎉
