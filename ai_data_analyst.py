import streamlit as st
import pandas as pd
import duckdb
import tempfile
import os
import re
import csv  # <-- Missing import added here
from groq import Groq

def sanitize_table_name(filename):
    """Convert filename to valid SQL table name"""
    name = os.path.splitext(filename)[0]  # Remove extension
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)  # Replace special chars
    name = name.strip('_')  # Remove leading/trailing underscores
    if not name[0].isalpha():
        name = 't_' + name  # Add prefix if starts with number
    return name.lower()

def preprocess_and_save(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8')
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
            
        # Clean column names for SQL compatibility
        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col).strip() for col in df.columns]
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)  # Requires csv module
            
        return temp_path, df.columns.tolist(), df, sanitize_table_name(file.name)
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None, None

# ... rest of the code remains the same ...

def extract_sql_query(response):
    sql_match = re.search(r"```sql\n(.*?)\n```", response, re.DOTALL)
    return sql_match.group(1).strip() if sql_match else None

def validate_sql_query(sql_query, valid_columns, table_name):
    # Check table name exists in query
    if table_name.lower() not in sql_query.lower():
        return False, f"Table '{table_name}' not referenced in query"
    
    # Extract columns ignoring aliases and functions
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

# Streamlit app
st.title("📊🔥 Data Query Analyst")

# Fetch Groq API key from Streamlit secrets
groq_key = st.secrets["GROQ_API_KEY"]

if uploaded_file := st.file_uploader("Upload data file", type=["csv", "xlsx"]):
    temp_path, columns, df, table_name = preprocess_and_save(uploaded_file)
    
    if temp_path and columns and df is not None and table_name:
        st.write("Preview:")
        st.dataframe(df.head(3))
        
        conn = duckdb.connect(':memory:')
        conn.register(table_name, df)
        
        st.markdown(f"**Available Columns 📄 in `{table_name}`:**")
        st.write([col for col in columns])

        query = st.text_area("Ask a question about the data:", 
                           placeholder="e.g., Show distribution of companies by country")

        if st.button("Analyze") and query:
            with st.spinner("Generating insights..."):
                try:
                    groq_client = Groq(api_key=groq_key)
                    prompt = f"""
                    You are a SQL expert analyzing {table_name} with columns: {columns}.
                    Generate a DuckDB-compatible SQL query that answers: "{query}".
                    - Use table name: {table_name}
                    - Use ONLY the provided columns
                    - Return ONLY the SQL query in ```sql blocks
                    - Avoid column aliases with spaces
                    """
                    
                    response = groq_client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="mixtral-8x7b-32768",
                    )
                    
                    raw_sql = extract_sql_query(response.choices[0].message.content)
                    
                    if raw_sql:
                        st.subheader("Generated Analysis")
                        st.code(raw_sql, language="sql")
                        
                        is_valid, validation_msg = validate_sql_query(raw_sql, columns, table_name)
                        
                        if is_valid:
                            try:
                                result = conn.execute(raw_sql).fetchdf()
                                st.dataframe(result)
                                st.success("Query executed successfully!")
                            except Exception as e:
                                st.error(f"Execution error: {str(e)}")
                        else:
                            st.error(f"Validation failed: {validation_msg}")
                    else:
                        st.error("Could not extract valid SQL from response")
                        
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
