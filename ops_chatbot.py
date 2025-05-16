# ops_chatbot.py
import streamlit as st
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import openai
import os
openai.api_key = os.getenv("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")



# Load your SOP PDF file
sop_loader = PyPDFLoader("operations_sop.pdf")
sop_docs = sop_loader.load()

# Embed SOP content
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(sop_docs, embeddings)
retriever = vector_store.as_retriever()

# Setup LLM QA Chain
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Load CSVs for inventory and order tracking
orders_df = pd.read_csv("orders_detailed.csv")
inventory_df = pd.read_csv("inventory_detailed.csv")

# --- UTILITY FUNCTIONS ---
def track_order(order_id):
    row = orders_df[orders_df["order_id"].astype(str) == str(order_id)]
    if not row.empty:
        return f"Order {order_id} is currently '{row.iloc[0]['status']}' and was placed on {row.iloc[0]['date']}."
    return "Order not found. Please check the order ID."

def check_stock(product):
    row = inventory_df[inventory_df["product_name"].str.lower() == product.lower()]
    if not row.empty:
        qty = row.iloc[0]['qty']
        return f"{product} has {qty} items in stock."
    return "Product not found in inventory."

def is_order_query(user_input):
    return "order" in user_input.lower() and any(char.isdigit() for char in user_input)

def is_inventory_query(user_input):
    return "stock" in user_input.lower() or "inventory" in user_input.lower()

# --- STREAMLIT APP ---
st.title("🛠️ OpsBot – Operations Chat Assistant")
user_query = st.text_input("Ask something about orders, inventory, or SOPs:")

if user_query:
    response = ""
    if is_order_query(user_query):
        words = user_query.split()
        order_id = next((w for w in words if w.isdigit()), None)
        if order_id:
            response = track_order(order_id)
        else:
            response = "Please include a valid order ID."

    elif is_inventory_query(user_query):
        # crude product guess (could be improved)
        words = user_query.lower().split()
        product_names = inventory_df["product_name"].str.lower().tolist()
        matched = next((word for word in words if word in product_names), None)
        if matched:
            response = check_stock(matched)
        else:
            response = "Please specify a known product."

    else:
        response = qa_chain.run(user_query)

    st.write(response)
