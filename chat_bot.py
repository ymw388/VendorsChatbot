# chat_bot.py

import os
from dotenv import load_dotenv
import boto3

# --- LangChain / Bedrock imports ---
from langchain_community.llms.bedrock import Bedrock
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent

# Load .env (if present) so boto3 can pick up AWS creds or profile
load_dotenv()

# --- AWS Bedrock client setup ---
# If you're on EC2 with an attached IAM role, no creds are needed in .env.
BEDROCK_REGION = os.getenv("AWS_REGION", "us-west-2")
bedrock_client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

# Wrap Bedrock in LangChain LLM interface
llm = Bedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    client=bedrock_client,
    temperature=0.1,
)

# --- SQLite database setup ---
DB_PATH = "spend_data.db"
# Explicitly expose only these tables to the LLM
TABLES = ["SLO_DATA", "Totals", "Copy", "Original", "Sub_Data", "DVBE_SB_MB"]
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}", include_tables=TABLES)

# --- Build the SQL Agent ---
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    handle_parsing_errors=True,   # retry on bad LLM output
)

# --- Chat loop ---
print("🤖 Welcome to the Spend Data Chatbot!")
print("Ask any question about your spend_data.db (type 'exit' or Ctrl+C to quit).")

try:
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("👋 Goodbye!")
            break

        # Run through the LangChain SQL agent
        response = agent_executor.run(user_input)
        print(f"\n🤖 {response}")

except KeyboardInterrupt:
    print("\n👋 Goodbye!")
