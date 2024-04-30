import os
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI, OpenAI
from langchain.agents import create_sql_agent, AgentExecutor

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = "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"

# Initialize the ChatOpenAI language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Database credentials
db_user = "mediyoga"
db_password = "mediyoga2023"
db_host = "3.111.133.67"
db_name = "Medi_Yoga"


mysql_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"


db = SQLDatabase.from_uri(mysql_uri)


toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

agent_executor.run("show me all the appointments today?")
