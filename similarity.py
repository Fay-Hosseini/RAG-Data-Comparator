import pandas as pd
import xml.etree.ElementTree as et
import json

from sklearn.metrics.pairwise import cosine_similarity

# Convert both to JSON-like texts for embeddings

# Excel
df = pd.read_excel("employees.xlsx")
excel_data = [row.to_json() for idx, row in df.iterrows()]
#print(excel_data)


# XAML
tree = et.parse("ui.xaml")
root = tree.getroot()
xaml_data = [json.dumps({"tag": e.tag, "attrib": e.attrib, "text": e.text}) for e in root.iter()]
#print(xaml_data)

# Convert text to numbers (vectors) so the model can search quickly.

from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

excel_embeddings = embed_model.encode(excel_data)
xaml_embeddings = embed_model.encode(xaml_data)

#print(excel_embeddings)





# Cosine similarity measures the angle between vectors: closer vectors = more similar meaning.
# This is the foundation of RAG retrieval: the system finds embeddings closest to your query.

excel_example = excel_data[0]  # '{"EmployeeId":101,"Name":"Alice","Role":"Developer"}'
xaml_example = xaml_data[6]    # '{"tag": "TextBox", "attrib": {"Name": "Name", "Content": "Alice"}, "text": null}'
print(xaml_example)

excel_vec = embed_model.encode([excel_example])
xaml_vec = embed_model.encode([xaml_example])

similarity = cosine_similarity(excel_vec, xaml_vec)     #similarity is a matrix of pairwise similarities
print(similarity)
print("Cosine similarity:", similarity[0][0])           #similarity[0][0] is the score for the first pair of vectors you compared.





