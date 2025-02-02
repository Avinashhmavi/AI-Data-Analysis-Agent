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
from wordcloud import WordCloud
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

# Function to create various charts with dynamic column selection
def create_chart(df, chart_type):
    st.subheader(f"📊 {chart_type}")
    columns = df.columns.tolist()
    
    if chart_type == "Bar Chart":
        x_col = st.selectbox("Select X-axis column", columns, index=0)
        y_col = st.selectbox("Select Y-axis column", columns, index=1%len(columns))
        fig = px.bar(df, x=x_col, y=y_col)
        st.plotly_chart(fig)
    
    elif chart_type == "Stacked Bar Chart":
        x_col = st.selectbox("Select X-axis column", columns, index=0)
        y_cols = st.multiselect("Select Y-axis columns", columns)
        if y_cols:
            fig = px.bar(df, x=x_col, y=y_cols, barmode="stack")
            st.plotly_chart(fig)
    
    elif chart_type == "Grouped Bar Chart":
        x_col = st.selectbox("Select X-axis column", columns, index=0)
        y_cols = st.multiselect("Select Y-axis columns", columns)
        if y_cols:
            fig = px.bar(df, x=x_col, y=y_cols, barmode="group")
            st.plotly_chart(fig)
    
    elif chart_type == "Line Chart":
        x_col = st.selectbox("Select X-axis column", columns, index=0)
        y_col = st.selectbox("Select Y-axis column", columns, index=1%len(columns))
        fig = px.line(df, x=x_col, y=y_col)
        st.plotly_chart(fig)
    
    elif chart_type == "Area Chart":
        x_col = st.selectbox("Select X-axis column", columns, index=0)
        y_col = st.selectbox("Select Y-axis column", columns, index=1%len(columns))
        fig = px.area(df, x=x_col, y=y_col)
        st.plotly_chart(fig)
    
    elif chart_type == "Stacked Area Chart":
        x_col = st.selectbox("Select X-axis column", columns, index=0)
        y_cols = st.multiselect("Select Y-axis columns", columns)
        if y_cols:
            fig = px.area(df, x=x_col, y=y_cols)
            st.plotly_chart(fig)
    
    elif chart_type == "Pie Chart":
        names_col = st.selectbox("Select Categories column", columns, index=0)
        values_col = st.selectbox("Select Values column", columns, index=1%len(columns))
        fig = px.pie(df, names=names_col, values=values_col)
        st.plotly_chart(fig)
    
    elif chart_type == "Donut Chart":
        names_col = st.selectbox("Select Categories column", columns, index=0)
        values_col = st.selectbox("Select Values column", columns, index=1%len(columns))
        fig = px.pie(df, names=names_col, values=values_col, hole=0.4)
        st.plotly_chart(fig)
    
    elif chart_type == "Histogram":
        x_col = st.selectbox("Select Column", columns, index=0)
        fig = px.histogram(df, x=x_col)
        st.plotly_chart(fig)
    
    elif chart_type == "Box Plot":
        y_cols = st.multiselect("Select Columns", columns)
        if y_cols:
            fig = px.box(df, y=y_cols)
            st.plotly_chart(fig)
    
    elif chart_type == "Violin Plot":
        y_col = st.selectbox("Select Y-axis column", columns, index=0)
        x_col = st.selectbox("Select X-axis column (optional)", [None] + columns)
        fig = px.violin(df, y=y_col, x=x_col)
        st.plotly_chart(fig)
    
    elif chart_type == "Scatter Plot":
        x_col = st.selectbox("Select X-axis column", columns, index=0)
        y_col = st.selectbox("Select Y-axis column", columns, index=1%len(columns))
        color_col = st.selectbox("Select Color column (optional)", [None] + columns)
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
        st.plotly_chart(fig)
    
    elif chart_type == "Bubble Chart":
        x_col = st.selectbox("Select X-axis column", columns, index=0)
        y_col = st.selectbox("Select Y-axis column", columns, index=1%len(columns))
        size_col = st.selectbox("Select Size column", columns, index=2%len(columns))
        fig = px.scatter(df, x=x_col, y=y_col, size=size_col)
        st.plotly_chart(fig)
    
    elif chart_type == "Heatmap":
        x_col = st.selectbox("Select X-axis column", columns, index=0)
        y_col = st.selectbox("Select Y-axis column", columns, index=1%len(columns))
        z_col = st.selectbox("Select Values column", columns, index=2%len(columns))
        fig = px.density_heatmap(df, x=x_col, y=y_col, z=z_col)
        st.plotly_chart(fig)
    
    elif chart_type == "Candlestick Chart":
        date_col = st.selectbox("Select Date column", columns)
        open_col = st.selectbox("Select Open column", columns)
        high_col = st.selectbox("Select High column", columns)
        low_col = st.selectbox("Select Low column", columns)
        close_col = st.selectbox("Select Close column", columns)
        fig = go.Figure(data=[go.Candlestick(
            x=df[date_col], open=df[open_col],
            high=df[high_col], low=df[low_col],
            close=df[close_col])])
        st.plotly_chart(fig)
    
    elif chart_type == "Treemap":
        path_cols = st.multiselect("Select Hierarchy columns", columns)
        value_col = st.selectbox("Select Value column", columns)
        if path_cols and value_col:
            fig = px.treemap(df, path=path_cols, values=value_col)
            st.plotly_chart(fig)
    
    elif chart_type == "Sankey Diagram":
        source_col = st.selectbox("Select Source column", columns)
        target_col = st.selectbox("Select Target column", columns)
        value_col = st.selectbox("Select Value column", columns)
        fig = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=20),
            link=dict(
                source=df[source_col].astype('category').cat.codes,
                target=df[target_col].astype('category').cat.codes,
                value=df[value_col]
            )
        ))
        st.plotly_chart(fig)
    
    elif chart_type == "Word Cloud":
        text_col = st.selectbox("Select Text column", columns)
        text = ' '.join(df[text_col].astype(str))
        wordcloud = WordCloud().generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)
    
    # Add more chart types following the same pattern

# Streamlit app UI
st.title("📊 AI-Powered Data Analyst & Visualization")

with st.sidebar:
    st.header("API Configuration")
    groq_key = st.text_input("Groq API Key:", type="password")

chart_options = [
    "Bar Chart", "Stacked Bar Chart", "Grouped Bar Chart",
    "Line Chart", "Area Chart", "Stacked Area Chart",
    "Pie Chart", "Donut Chart", "Histogram", "Box Plot",
    "Violin Plot", "Scatter Plot", "Bubble Chart", "Heatmap",
    "Candlestick Chart", "Treemap", "Sankey Diagram", "Word Cloud"
]

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
                    Suggest a best-fit chart type from: {chart_options}.
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

                                chart_type = st.selectbox("Select Visualization Type", chart_options)
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
