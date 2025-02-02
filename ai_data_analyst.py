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
            df = pd.read_excel(file, engine='openpyxl')  # Ensure Excel support
        
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

# Extract chart type suggestion from AI response
def extract_chart_type(response):
    match = re.search(r'Chart Type: (.*?)\n', response)
    return match.group(1).strip() if match else "line"

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

        query = st.text_area("💬 Ask a question about the data:", placeholder="e.g., Show stock price trend with moving average")

        if st.button("🔍 Analyze"):
            with st.spinner("Analyzing... Please wait..."):
                try:
                    groq_client = Groq(api_key=groq_key)
                    prompt = f"""
                    You are an AI data analyst with {table_name} containing columns: {columns}.
                    Generate a DuckDB-compatible SQL query that answers: "{query}".
                    Suggest an appropriate chart type (line, bar, histogram, scatter, pie, candlestick).
                    - Use table: {table_name}
                    - Use only given columns
                    - Return SQL inside ```sql``` blocks
                    - Mention chart type as "Chart Type: XYZ"
                    """
                    
                    response = groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="mixtral-8x7b-32768",
                    )
                    
                    raw_sql = extract_sql_query(response.choices[0].message.content)
                    chart_type = extract_chart_type(response.choices[0].message.content)
                    
                    if raw_sql:
                        st.subheader("🔍 AI-Generated SQL Query")
                        st.code(raw_sql, language="sql")
                        
                        is_valid, validation_msg = validate_sql_query(raw_sql, columns, table_name)
                        
                        if is_valid:
                            try:
                                result = conn.execute(raw_sql).fetchdf()
                                st.dataframe(result)
                                
                                # Chart Visualization
                                st.subheader("📊 AI-Generated Visualization")
                                
                                if chart_type == "candlestick" and len(result.columns) >= 5:
                                    fig = go.Figure(data=[go.Candlestick(
                                        x=pd.to_datetime(result[result.columns[0]]),
                                        open=result[result.columns[1]],
                                        high=result[result.columns[2]],
                                        low=result[result.columns[3]],
                                        close=result[result.columns[4]],
                                    )])
                                    fig.update_layout(title="Candlestick Chart", xaxis_title="Date", yaxis_title="Price")
                                    st.plotly_chart(fig)
                                
                                elif chart_type == "moving average" and len(result.columns) >= 2:
                                    period = st.slider("Select Moving Average Window:", min_value=2, max_value=50, value=10)
                                    result['Moving_Avg'] = result[result.columns[1]].rolling(window=period).mean()
                                    fig = px.line(result, x=result.columns[0], y=[result.columns[1], 'Moving_Avg'], title="Moving Average")
                                    st.plotly_chart(fig)

                                elif chart_type == "histogram":
                                    fig = px.histogram(result, x=result.columns[0], title="Histogram")
                                    st.plotly_chart(fig)

                                elif chart_type == "scatter" and len(result.columns) >= 2:
                                    fig = px.scatter(result, x=result.columns[0], y=result.columns[1], title="Scatter Plot")
                                    st.plotly_chart(fig)

                                elif chart_type == "pie" and len(result.columns) >= 2:
                                    fig = px.pie(result, names=result.columns[0], values=result.columns[1], title="Pie Chart")
                                    st.plotly_chart(fig)

                                else:  # Default to line chart
                                    fig = px.line(result, x=result.columns[0], y=result.columns[1], title="Line Chart")
                                    st.plotly_chart(fig)

                                st.success("✅ Query executed successfully!")
                            except Exception as e:
                                st.error(f"❌ Execution error: {str(e)}")
                        else:
                            st.error(f"❌ Validation failed: {validation_msg}")
                    else:
                        st.error("⚠️ AI did not return a valid SQL query.")
                        
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
