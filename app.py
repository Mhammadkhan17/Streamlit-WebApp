import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # for more advanced plots
import plotly.express as px # for interactive plots
import google.generativeai as genai
import os

# API key setup - place at the very beginning
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model_name = 'models/gemini-2.0-flash-thinking-exp-01-21' # Specified model

st.title("Enhanced Data Explorer with Gemini Chatbot")

# Sidebar for controls
st.sidebar.header("Data Loading and Configuration")

# File Upload
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            st.stop() # Stop execution if unsupported file type
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Store df in session state to be accessible by chatbot
    st.session_state['df'] = df

    # --- Data Overview ---
    st.header("Data Overview")

    with st.expander("Show Raw Data"):
        st.dataframe(df)

    with st.expander("Data Summary"):
        st.write("Descriptive Statistics:")
        st.write(df.describe())

        st.write("\nData Types:")
        st.write(df.dtypes)

        st.write("\nMissing Values:")
        st.write(df.isnull().sum())

        st.write(f"\nNumber of Rows: {df.shape[0]}")
        st.write(f"Number of Columns: {df.shape[1]}")


    # --- Data Filtering ---
    st.header("Data Filtering")
    filter_container = st.container() # Use a container to group filters

    with filter_container:
        num_filters = st.number_input("Number of filters", min_value=0, max_value=5, value=0) # Limit to 5 for UI clarity
        filters = []
        for i in range(num_filters):
            col_to_filter = st.selectbox(f"Filter Column {i+1}", df.columns)
            filter_type = st.selectbox(f"Filter Type {i+1} for {col_to_filter}", ["equals", "greater than", "less than", "contains"])
            filter_value = st.text_input(f"Value for filter {i+1} on {col_to_filter}")
            filters.append({'column': col_to_filter, 'type': filter_type, 'value': filter_value})

        filtered_df = df.copy() # Start with a copy to avoid modifying original
        for filter_item in filters:
            col = filter_item['column']
            filter_type = filter_item['type']
            value = filter_item['value']

            if value: # Only apply filter if a value is entered
                try: # Handle potential type errors during comparison
                    if filter_type == "equals":
                        filtered_df = filtered_df[filtered_df[col] == value]
                    elif filter_type == "greater than":
                        filtered_df = filtered_df[filtered_df[col] > pd.to_numeric(value)] # Convert to numeric for comparison
                    elif filter_type == "less than":
                        filtered_df = filtered_df[filtered_df[col] < pd.to_numeric(value)] # Convert to numeric
                    elif filter_type == "contains":
                        filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(value, case=False, na=False)] # String contains, handle NaN
                except Exception as e:
                    st.error(f"Error applying filter on column '{col}': {e}. Please check filter value type.")

        st.subheader("Filtered Data")
        st.write(filtered_df)
        st.session_state['filtered_df'] = filtered_df # Store filtered_df in session state

    # --- Data Visualization ---
    st.header("Data Visualization")

    plot_type = st.selectbox("Select Plot Type", ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot"])
    columns_for_plot = df.columns.tolist()

    if plot_type in ["Line Chart", "Scatter Plot", "Bar Chart"]:
        x_column = st.selectbox("Select X-axis column", columns_for_plot, key="x_col")
        y_column = st.selectbox("Select Y-axis column", columns_for_plot, key="y_col")
        if st.button("Generate Plot", key="plot_button_xy"):
            st.subheader(f"{plot_type} of {y_column} vs {x_column}")
            try:
                if plot_type == "Line Chart":
                    fig = px.line(filtered_df, x=x_column, y=y_column, title=f"Line Chart: {y_column} vs {x_column}")
                elif plot_type == "Scatter Plot":
                    fig = px.scatter(filtered_df, x=x_column, y=y_column, title=f"Scatter Plot: {y_column} vs {x_column}")
                elif plot_type == "Bar Chart":
                    fig = px.bar(filtered_df, x=x_column, y=y_column, title=f"Bar Chart: {y_column} vs {x_column}")
                st.plotly_chart(fig) # Use plotly for interactive charts
            except Exception as e:
                st.error(f"Error generating plot: {e}. Please ensure selected columns are suitable for {plot_type}.")

    elif plot_type in ["Histogram", "Box Plot"]:
        column_to_plot = st.selectbox("Select column to plot", columns_for_plot, key="hist_box_col")
        if st.button("Generate Plot", key="plot_button_single"):
            st.subheader(f"{plot_type} of {column_to_plot}")
            try:
                if plot_type == "Histogram":
                    fig = px.histogram(filtered_df, x=column_to_plot, title=f"Histogram of {column_to_plot}")
                elif plot_type == "Box Plot":
                    fig = px.box(filtered_df, y=column_to_plot, title=f"Box Plot of {column_to_plot}") # y-axis for box plot
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error generating plot: {e}. Please ensure selected column is suitable for {plot_type}.")

    # # --- Download Filtered Data ---
    # st.header("Download Data")
    # st.download_button(
    #     label="Download Filtered Data as CSV",
    #     data=filtered_df.to_csv(index=False),
    #     file_name="filtered_data.csv",
    #     mime="text/csv"
    # )

    # --- Gemini Chatbot Section ---
    st.header("Chatbot for Data Analysis")

    if 'df' in st.session_state: # Only show chatbot if dataframe exists
        df = st.session_state['df'] # Retrieve dataframe from session state

        available_models = [model_name] # Use the specified model only
        selected_model_name = model_name # Directly select the specified model
        model = genai.GenerativeModel(selected_model_name)

        # Initialize chat history in Streamlit session state
        if "chat_history_data" not in st.session_state:
            st.session_state.chat_history_data = []

        # Display existing chat messages from history
        for message in st.session_state.chat_history_data:
            with st.chat_message(message["role"]):
                st.markdown(message["parts"][0])

        if prompt := st.chat_input("Ask questions about your data:"):
            st.chat_message("user").markdown(prompt)

            # Prepare context for the model - include column names and data summary
            context = f"You are a helpful chatbot assistant specialized in understanding and analyzing tabular data. The user has uploaded a dataset. Here is a summary of the dataset to help you understand it:\n\n"
            context += "Column Names: " + ", ".join(df.columns.tolist()) + "\n\n"
            context += "Descriptive Statistics:\n" + df.describe().to_string() + "\n\n"
            context += "Answer questions based on this data. If the question is not related to the data, politely say you can only answer questions about the uploaded dataset.\n\n"

            full_prompt = context + "User Question: " + prompt

            st.session_state.chat_history_data.append({"role": "user", "parts": [prompt]})

            try:
                response = model.generate_content([full_prompt]) # Pass full prompt with context
                bot_response = response.text
            except Exception as e:
                bot_response = f"An error occurred: {e}"

            st.chat_message("assistant").markdown(bot_response)
            st.session_state.chat_history_data.append({"role": "model", "parts": [bot_response]})

    else:
        st.write("Upload a CSV or Excel file to enable the chatbot.")

else:
    st.write("Waiting for file upload...")