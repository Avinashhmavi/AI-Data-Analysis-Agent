import streamlit as st
import pandas as pd
import duckdb
import tempfile
import os
import re
import csv
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

# Function to clean table names
def sanitize_table_name(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name).strip('_')
    return 't_' + name.lower() if not name[0].isalpha() else name.lower()

# Function to process uploaded file
def preprocess_and_save(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl') 
        
        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col).strip() for col in df.columns]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
            
        return temp_path, df.columns.tolist(), df, sanitize_table_name(file.name)
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None, None

# Extract SQL query from AI response
def extract_sql_query(response):
    sql_match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL)
    return sql_match.group(1).strip() if sql_match else None

# Validate AI-generated SQL query
def validate_sql_query(sql_query, valid_columns, table_name):
    if table_name.lower() not in sql_query.lower():
        return False, f"Table '{table_name}' not referenced in query"
    
    pattern = re.compile(r'\b(\w+)\b(?=\s*[,\)]|\s+FROM|\s+WHERE)')
    used_columns = set()
    
    for match in pattern.finditer(sql_query):
        word = match.group(1).lower()
        if word in map(str.lower, valid_columns):
            used_columns.add(word)
    
    invalid = [col for col in used_columns if col.lower() not in map(str.lower, valid_columns)]
    if invalid:
        return False, f"Invalid columns: {', '.join(invalid)}"
    
    return True, ""

# Function to create various charts based on data
def create_chart(df, chart_type):
    st.subheader(f"📊 {chart_type} Chart")

    if chart_type == "Bar":
        fig = px.bar(df, x=df.columns[0], y=df.columns[1], title="Bar Chart")
    elif chart_type == "Stacked Bar":
        fig = px.bar(df, x=df.columns[0], y=df.columns[1:], title="Stacked Bar Chart", barmode="stack")
    elif chart_type == "Grouped Bar":
        fig = px.bar(df, x=df.columns[0], y=df.columns[1:], title="Grouped Bar Chart", barmode="group")
    elif chart_type == "Line":
        fig = px.line(df, x=df.columns[0], y=df.columns[1], title="Line Chart")
    elif chart_type == "Area":
        fig = px.area(df, x=df.columns[0], y=df.columns[1], title="Area Chart")
    elif chart_type == "Stacked Area":
        fig = px.area(df, x=df.columns[0], y=df.columns[1:], title="Stacked Area Chart")
    elif chart_type == "Pie":
        fig = px.pie(df, names=df.columns[0], values=df.columns[1], title="Pie Chart")
    elif chart_type == "Donut":
        fig = px.pie(df, names=df.columns[0], values=df.columns[1], hole=0.4, title="Donut Chart")
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=df.columns[0], title="Histogram")
    elif chart_type == "Box Plot":
        fig = px.box(df, y=df.columns[1:], title="Box Plot")
    elif chart_type == "Scatter":
        fig = px.scatter(df, x=df.columns[0], y=df.columns[1], title="Scatter Plot")
    elif chart_type == "Bubble":
        fig = px.scatter(df, x=df.columns[0], y=df.columns[1], size=df[df.columns[2]], title="Bubble Chart")
    elif chart_type == "Heatmap":
        fig = sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        st.pyplot(fig.figure)
        return
    elif chart_type == "Violin":
        fig = px.violin(df, y=df.columns[1], title="Violin Plot")
    elif chart_type == "Candlestick":
        fig = go.Figure(data=[go.Candlestick(
            x=df[df.columns[0]],
            open=df[df.columns[1]],
            high=df[df.columns[2]],
            low=df[df.columns[3]],
            close=df[df.columns[4]]
        )])
        fig.update_layout(title="Candlestick Chart")
    else:
        st.error("Unsupported chart type selected.")
        return
    
    st.plotly_chart(fig)

# Streamlit app UI
st.title("📊 AI-Powered Data Analyst & Visualization")

with st.sidebar:
    st.header("API Configuration")
    groq_key = st.text_input("Groq API Key:", type="password")

if uploaded_file := st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"]):
    temp_path, columns, df, table_name = preprocess_and_save(uploaded_file)
    
    if temp_path and columns and df is not None and table_name:
        st.write("📌 **Data Preview:**")
        st.dataframe(df.head(5))
        
        conn = duckdb.connect(':memory:')
        conn.register(table_name, df)
        
        st.markdown(f"**📌 Available Columns in `{table_name}`:**")
        st.write(columns)

        query = st.text_area("💬 Ask a question about the data:", placeholder="e.g., Show sales trends over time")

        if st.button("🔍 Analyze"):
            with st.spinner("Analyzing... Please wait..."):
                try:
                    groq_client = Groq(api_key=groq_key)
                    prompt = f"""
                    Generate a DuckDB SQL query for {table_name} with columns: {columns}.
                    Suggest a best-fit chart type (Bar, Line, Pie, Heatmap, Scatter, etc.).
                    """
                    
                    response = groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="mixtral-8x7b-32768",
                    )
                    
                    raw_sql = extract_sql_query(response.choices[0].message.content)
                    
                    if raw_sql:
                        st.subheader("🔍 AI-Generated SQL Query")
                        st.code(raw_sql, language="sql")
                        
                        is_valid, validation_msg = validate_sql_query(raw_sql, columns, table_name)
                        
                        if is_valid:
                            try:
                                result = conn.execute(raw_sql).fetchdf()
                                st.dataframe(result)

                                chart_type = st.selectbox("Select Visualization Type", [
                                    "Bar", "Line", "Pie", "Histogram", "Scatter", "Bubble", "Box Plot", "Violin", "Heatmap", "Candlestick"
                                ])

                                create_chart(result, chart_type)

                                st.success("✅ Query executed successfully!")
                            except Exception as e:
                                st.error(f"❌ Execution error: {str(e)}")
                        else:
                            st.error(f"❌ Validation failed: {validation_msg}")
                    else:
                        st.error("⚠️ AI did not return a valid SQL query.")
                        
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
