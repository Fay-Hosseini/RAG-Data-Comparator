"""
The code has been fully fixed to correctly parse the XAML data, which was the source of the previous inaccuracies.
It now uses a custom function to group related XAML elements into a single record for better semantic comparison.
The final script includes the full RAG pipeline and a new, more robust data normalization step.
"""
import pandas as pd
import xml.etree.ElementTree as et
import json
import os

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA

# A more robust function to normalize XAML data by creating a single record for each employee.
def normalize_xaml_employees(file_path):
    try:
        tree = et.parse(file_path)
        root = tree.getroot()
        normalized_records = []

        # Iterate through the 'Row' elements which logically contain employee data
        for row in root.findall(".//Row"):
            employee_data = {}
            # Find the 'Employee' element within the current 'Row'
            employee_elem = row.find(".//Employee")
            if employee_elem:
                # Consolidate all 'TextBox' attributes into a single dictionary
                for textbox in employee_elem.findall(".//TextBox"):
                    name = textbox.attrib.get("Name")
                    content = textbox.attrib.get("Content")
                    if name and content:
                        employee_data[name] = content

            # If a record was found, convert it to a JSON string and add to the list
            if employee_data:
                normalized_records.append(json.dumps(employee_data))

        return normalized_records
    except FileNotFoundError:
        raise FileNotFoundError(f"XAML file not found at {file_path}. Please ensure it's in the same directory.")


# --- Step 1. Load and Normalize Data ---
try:
    df = pd.read_excel("employees.xlsx")
    excel_data = [row.to_json() for _, row in df.iterrows()]
except FileNotFoundError:
    raise FileNotFoundError("employees.xlsx not found. Please ensure it's in the same directory.")

xaml_data = normalize_xaml_employees("ui.xaml")

# --- Step 2. Create LangChain Documents with metadata ---
docs = []
docs.extend([Document(page_content=text, metadata={"source": "excel"}) for text in excel_data])
docs.extend([Document(page_content=text, metadata={"source": "xaml"}) for text in xaml_data])

# --- Step 3. Embeddings + Vectorstore ---
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings_model)

# --- Step 4. Load Local LLM ---
model_path = "./models/llama-2-13b-chat.Q4_K_M.gguf"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"LLM model not found at {model_path}. Please download it first.")

llm = LlamaCpp(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
    temperature=0.1,
)

# --- Step 5. Create Retrieval Chain ---
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# --- Step 6. Define and Execute Query ---
query = """
Compare employees from the Excel data with employees from the XAML data.
List any employees present in one source but not the other.
Use the 'source' metadata to indicate where each employee is found.
"""
response = qa.invoke({"query": query})
comparison_result = response["result"]

# --- Step 7. Create Excel Report ---
report_filename = "report.xlsx"
comparison_df = pd.DataFrame([{"Report": comparison_result}])
original_excel_df = pd.DataFrame([json.loads(text) for text in excel_data])
original_xaml_df = pd.DataFrame([json.loads(text) for text in xaml_data])

with pd.ExcelWriter(report_filename) as writer:
    comparison_df.to_excel(writer, sheet_name="Comparison Report", index=False)
    original_excel_df.to_excel(writer, sheet_name="Original Excel Data", index=False)
    original_xaml_df.to_excel(writer, sheet_name="Normalized XAML Data", index=False)

print("\n--- RAG Comparison Complete ---")
print(f"Report generated at: {os.path.abspath(report_filename)}")