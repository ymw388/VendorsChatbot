import os
from dotenv import load_dotenv
import boto3

# LangChain imports
from langchain_community.chat_models.bedrock import BedrockChat
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent

# Load environment variables (if using .env)
load_dotenv()

# --- AWS Bedrock client setup ---
REGION = os.getenv("AWS_REGION", "us-west-2")
bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)

# --- LangChain LLM setup ---
llm = BedrockChat(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    client=bedrock_client,
    temperature=0.1,
)

# --- SQLite database setup ---
DB_PATH = "spend_data.db"
TABLES = [
    "SLO_DATA",
    "Totals",
    "Copy",
    "Original",
    "Sub_Data",
    "DVBE_SB_MB",
]
db = SQLDatabase.from_uri(
    f"sqlite:///{DB_PATH}",
    include_tables=TABLES
)

# --- Create SQL Agent ---
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    handle_parsing_errors=True,
)

# --- Chatbot Loop ---
def main():
    print("🤖 Welcome to the Spend Data Chatbot!")
    print("Ask a question about your spend_data.db (type 'exit' to quit).")

    try:
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("👋 Goodbye!")
                break

            response = agent_executor.run(user_input)
            print(f"\n🤖 {response}")

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
