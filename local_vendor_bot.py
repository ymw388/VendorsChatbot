import os
import time
import pandas as pd
from langchain_core.documents import Document
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

# AWS region for Bedrock
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

# Load CSV with a tolerant encoding
df = pd.read_csv("vendor_certifications.csv", encoding="latin1")

# Convert rows to LangChain Documents
documents = []
for idx, row in df.iterrows():
    text = "\n".join(f"{col}: {row[col]}" for col in df.columns)
    documents.append(Document(page_content=text))

# Embed using Amazon Titan via Bedrock
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name=AWS_REGION
)

# Build a local FAISS vector index with simple retry
max_retries = 3
for attempt in range(max_retries):
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local("vendor_index")
        break
    except Exception as e:
        if "throttl" in str(e).lower() and attempt < max_retries - 1:
            wait_time = (2 ** attempt) * 5
            print(f"Rate limited. Waiting {wait_time} seconds before retry {attempt + 2}...")
            time.sleep(wait_time)
        else:
            raise e

# CLI loop
print("🛍️  Small Business Search Bot (Powered by Titan Embeddings)\n")
while True:
    user_input = input("🔍 What are you looking for? (or type 'exit')\n> ")
    if user_input.strip().lower() in ["exit", "quit", "bye"]:
        print("👋 Goodbye!")
        break

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    results = retriever.invoke(user_input)

    print("\n✨ Top Recommendations:")
    for i, doc in enumerate(results, 1):
        print(f"\n#{i}:\n{doc.page_content}")
    print("\n" + "-" * 50)
