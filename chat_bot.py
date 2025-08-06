# chat_bot.py

import os
from dotenv import load_dotenv
from langchain.chat_models import ChatBedrock
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
import boto3

# Load environment variables (optional if using IAM role on EC2)
load_dotenv()

# --- AWS Bedrock Setup ---
# Make sure the EC2 IAM role has Bedrock permissions, or set .env with keys
bedrock_client = boto3.client("bedrock-runtime", region_name="us-west-2")

# Choose a Bedrock model
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",  # Or switch to Titan/Mistral
    client=bedrock_client,
    temperature=0.1
)

# --- SQLite Database Setup ---
# Assumes spend_data.db is in the current directory
db_uri = "sqlite:///spend_data.db"
db = SQLDatabase.from_uri(db_uri)

# --- LangChain Agent Setup ---
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
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
