# chat_bot.py

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatBedrock
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
import boto3

# Load environment variables (optional if using IAM role on EC2)
load_dotenv()

# --- AWS Bedrock Setup ---
bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")

llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",  # Bedrock model
    client=bedrock_client,
    temperature=0.1
)

# --- SQLite Database Setup ---
db_uri = "sqlite:///spend_data.db"

# 👇 Explicitly specify the tables you want the LLM to know about
db = SQLDatabase.from_uri(
    db_uri,
    include_tables=["SLO_DATA", "Totals", "Copy", "Original", "Sub_Data", "DVBE_SB_MB"]
)

# --- LangChain Agent Setup ---
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    handle_parsing_errors=True  # 👈 key addition to fix your error
)

# --- Chatbot Loop ---
print("🤖 Welcome to the Spend Data Chatbot!")
print("Type your question about the data (or type 'exit' to quit).")

while True:
    user_input = input("\nYou: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    try:
        response = agent_executor.run(user_input)
        print(f"\n🤖 {response}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
