import pandas as pd
import streamlit as st
import re
import unicodedata
from ydata_profiling import ProfileReport
import tempfile
import os

# Sidebar for page navigation
st.set_page_config(page_title="Data Quality Rule Tester")

# Sidebar for page navigation
# page = st.sidebar.selectbox("Select Page", ["Guidance", "Profiling Report", "Single Rule Testing", "Lookup Rule Testing", "Composite Key Lookup"])

page = st.sidebar.radio(
    "📂 Navigation",
    [
        "Guidance",
        "Profiling Report",        
        "Single Rule Testing",
        "Lookup Rule Testing",
        "Composite Key Lookup",        
    ]
)

if page == "Single Rule Testing":
    st.title("🧪 Simple Data Quality Rule Tester")

    st.markdown("""
    Upload a CSV, write a Python lambda rule, and apply it to one column.

    **Example rule:**
    ```python
    lambda x: x > 0
    ```
    """)

    file = st.file_uploader("Upload your CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("### 📊 Preview of Data:")
        st.dataframe(df.head())

        selected_column = st.selectbox("Select a column to apply a rule to:", df.columns)
        rule_input = st.text_area("Write your data quality rule:", "lambda x: x is not None")

        if st.button("✅ Apply Rule"):
            try:
                local_env = {}
                exec(f"dq_rule = {rule_input}", {}, local_env)
                dq_rule = local_env.get("dq_rule")

                if not callable(dq_rule):
                    raise ValueError("Rule must be a callable lambda function.")

                def safe_apply(func, value):
                    try:
                        result = func(value)
                        if not isinstance(result, bool):
                            raise ValueError("Rule must return a boolean value.")
                        return result
                    except Exception:
                        return False

                result = df[selected_column].apply(lambda x: safe_apply(dq_rule, x))
                df.insert(0, 'DQ_Result', result)
                failed_rows = df[~result]

                st.success(f"✅ {result.sum()} passed, ❌ {len(failed_rows)} failed the rule.")
                st.write("### 📌 Full Results (with Highlight):")
                st.dataframe(df.style.applymap(lambda v: 'background-color: #ffcccc' if v is False else '', subset=['DQ_Result']))
            except Exception as e:
                st.error(f"Error applying rule: {e}")

elif page == "Lookup Rule Testing":
    st.title("🔍 Lookup Rule Tester")

    st.markdown("""
    Upload a main CSV and a lookup CSV. Then validate a column in the main file based on values in the lookup file.

    **Example rule:**
    ```python
    lambda x: x in lookup_list
    ```
    """)

    main_file = st.file_uploader("Upload main data CSV", key="main", type=["csv"])
    lookup_file = st.file_uploader("Upload lookup table CSV", key="lookup", type=["csv"])

    if main_file and lookup_file:
        df = pd.read_csv(main_file)
        lookup_df = pd.read_csv(lookup_file)

        st.write("### Main Data Preview:")
        st.dataframe(df.head())
        st.write("### Lookup Table Preview:")
        st.dataframe(lookup_df.head())

        main_column = st.selectbox("Select column to validate:", df.columns)
        lookup_column = st.selectbox("Select lookup column:", lookup_df.columns)

        rule_input = st.text_area("Write your rule (you can use 'lookup_list'):",
                                  "lambda x: x in lookup_list")

        if st.button("🔍 Apply Lookup Rule"):
            try:
                # Deep clean both lookup and main column
                lookup_list = (
                    lookup_df[lookup_column]
                    .dropna()
                    .apply(str)
                    .str.encode('ascii', 'ignore')
                    .str.decode('ascii')
                    .str.strip()
                    .tolist()
                )

                df[main_column] = (
                    df[main_column]
                    .apply(str)
                    .str.encode('ascii', 'ignore')
                    .str.decode('ascii')
                    .str.strip()
                )

                # Evaluate rule using lookup_list in scope
                dq_rule = eval(rule_input, {"lookup_list": lookup_list})

                if not callable(dq_rule):
                    raise ValueError("Rule must be a callable lambda function.")

                def safe_apply(func, value):
                    try:
                        result = func(value)
                        if not isinstance(result, bool):
                            raise ValueError("Rule must return a boolean value.")
                        return result
                    except Exception:
                        return False

                result = df[main_column].apply(lambda x: safe_apply(dq_rule, x))
                df.insert(0, 'DQ_Result', result)
                failed_rows = df[~result]

                passed = result.sum()
                failed = len(failed_rows)

                st.success(f"✅ {passed} passed, ❌ {failed} failed the rule.")
                st.write("### 📌 Full Results (with Highlight):")
                st.dataframe(df.style.applymap(lambda v: 'background-color: #ffcccc' if v is False else '', subset=['DQ_Result']))

                if passed == 0:
                    st.warning("⚠️ All rows failed. Here's a sample of what you're comparing:")
                    st.write("**Sample values in main column:**", df[main_column].dropna().unique()[:5].tolist())
                    st.write("**Sample values in lookup list:**", lookup_list[:5])
                    types_main = df[main_column].map(lambda x: str(type(x))).unique().tolist()
                    types_lookup = list(set([str(type(x)) for x in lookup_list]))
                    st.write("**Main column value types:**", types_main)
                    st.write("**Lookup list value types:**", types_lookup)

            except Exception as e:
                st.error(f"Error applying lookup rule: {e}")

elif page == "Composite Key Lookup":
    st.title("🔗 Composite Key Lookup Rule Tester")

    st.markdown("""
    Upload a main CSV and a lookup CSV. Then select multiple columns from each to match composite keys.
    The row will pass if the combination of values exists in the lookup table.
    """)

    main_file = st.file_uploader("Upload main data CSV", key="composite_main", type=["csv"])
    lookup_file = st.file_uploader("Upload lookup data CSV", key="composite_lookup", type=["csv"])

    if main_file and lookup_file:
        df_main = pd.read_csv(main_file)
        df_lookup = pd.read_csv(lookup_file)

        main_columns = st.multiselect("Select columns from main data (composite key):", df_main.columns)
        lookup_columns = st.multiselect("Select columns from lookup data (composite key):", df_lookup.columns)

        if main_columns and lookup_columns and len(main_columns) == len(lookup_columns):
            df_main_preview = df_main[main_columns].copy()
            df_lookup_preview = df_lookup[lookup_columns].copy()
            df_main_preview.columns = [f"main.{col}" for col in main_columns]
            df_lookup_preview.columns = [f"lookup.{col}" for col in lookup_columns]
            st.write("### 🔍 Side-by-Side Composite Key Preview:")
            st.dataframe(pd.concat([df_main_preview, df_lookup_preview], axis=1))

        if st.button("🔍 Validate Composite Keys"):
            if len(main_columns) != len(lookup_columns):
                st.error("Main and lookup column selections must have the same number of columns.")
            else:
                try:
                    def normalize(val):
                        return unicodedata.normalize("NFKD", str(val)).strip().lower()

                    def clean_composite(df, cols):
                        return df[cols].astype(str).apply(lambda row: '|'.join([normalize(v) for v in row]), axis=1)

                    df_main['__composite_key__'] = clean_composite(df_main, main_columns)
                    df_lookup['__composite_key__'] = clean_composite(df_lookup, lookup_columns)

                    lookup_keys = set(df_lookup['__composite_key__'])
                    df_main['DQ_Result'] = df_main['__composite_key__'].isin(lookup_keys)

                    st.success(f"✅ {df_main['DQ_Result'].sum()} passed, ❌ {(~df_main['DQ_Result']).sum()} failed the composite key lookup.")
                    st.write("### 📌 Full Results (Side-by-Side Composite Match):")

                    matched_lookup = pd.merge(
                        df_main[['__composite_key__']],
                        df_lookup[['__composite_key__'] + lookup_columns],
                        on='__composite_key__',
                        how='left'
                    )

                    result_df = pd.concat([
                        df_main[['DQ_Result', '__composite_key__'] + main_columns].add_prefix("main."),
                        matched_lookup[lookup_columns].add_prefix("lookup."),
                        matched_lookup[['__composite_key__']].rename(columns={'__composite_key__': 'lookup.__composite_key__'})
                    ], axis=1)

                    st.dataframe(result_df.style.applymap(
                        lambda v: 'background-color: #ffcccc' if v is False else '', subset=['main.DQ_Result']
                    ))

                except Exception as e:
                    st.error(f"Error during composite key validation: {e}")

elif page == "Profiling Report":

    st.title("📊 Data Profiling Report")

    uploaded_file = st.file_uploader("Upload a CSV file for profiling", type=["csv"], key="profiling_file")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())

        if st.button("🧪 Generate Profiling Report"):
            with st.spinner("Generating report..."):
                profile = ProfileReport(df, title="Profiling Report", explorative=True)
                output_path = os.path.join(os.getcwd(), "profiling_report.html")
                profile.to_file(output_path)
                with open(output_path, "r", encoding="utf-8") as f:
                    html = f.read()
                st.success("Report generated. Save or open from below.")
                st.download_button("💾 Download Report", data=html, file_name="profiling_report.html", mime="text/html")
                st.components.v1.html(html, height=1000, scrolling=True)


elif page == "Guidance":
    st.title("📘 Guidance: How to Use the App")

    st.markdown("""
                
    ### 📊 Profiling Report
    - Upload one CSV file.
    - Click "Generate Profiling Report".
    - View or download the full HTML report.
    
    ---
                
    ### 🧪 Single Rule Testing
    - Upload your main dataset as CSV.
    - Select the column you want to test.
    - Write a Python lambda rule using `x`, for example:

        - `lambda x: x is not None`
        - `lambda x: len(str(x)) > 0`
        - `lambda x: x.isdigit()`
        - `lambda x: '@' in x`  (for basic email check)

    - Apply the rule to get pass/fail results.

    ---

    ### 🔍 Lookup Rule Testing
    - Upload both a **main** and **lookup** CSV file.
    - Select a column from each file.
    - Write your rule using `x` and `lookup_list` (a list of values from lookup column):

        - `lambda x: x in lookup_list`
        - `lambda x: x.lower() in [i.lower() for i in lookup_list]`

    ---

    ### 🔗 Composite Key Lookup
    - Upload both files.
    - Select multiple columns from both files that form a composite key.
    - App will normalize and match combined values.
    - Matching is done like:

        - `CustomerID|FullName`

    --- 

    ✅ Common Lambda Rules:
    | Rule Description                      | Lambda Expression                                                   |
    |---------------------------------------|----------------------------------------------------------------------|
    | Not Null                              | `lambda x: x is not None`                                           |
    | Not Empty                             | `lambda x: str(x).strip() != ''`                                    |
    | Numeric Only                          | `lambda x: str(x).isdigit()`                                        |
    | Email Format (basic)                  | `lambda x: '@' in str(x)`                                           |
    | String Length > 5                     | `lambda x: len(str(x)) > 5`                                         |
    | Not Empty if Not Null                 | `lambda x: True if x is None else str(x).strip() != ''`            |
    | Starts with digit & numeric           | `lambda x: x.isdigit() if str(x)[0].isdigit() else False`          |
    | Only check len if string              | `lambda x: len(x) > 5 if isinstance(x, str) else False`            |
    | Valid domain if email format present  | `lambda x: x.endswith('@example.com') if '@' in str(x) else False` |

    Feel free to experiment with any valid Python `lambda` expression!
    """)

    st.markdown("""
    ---
    Made with ❤️ using Streamlit — customized by **Hussain Alhadab & Ibrahim Hassounah**
    """)