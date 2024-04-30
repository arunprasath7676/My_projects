# Import required modules
import sqlite3
import os
import pymysql
from sqlalchemy import create_engine,text
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.llms.openai import OpenAI
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

db_user = "mediyoga"
db_password = "mediyoga2023"
db_host = "3.111.133.67"
db_name = "Medi_Yoga"

mysql_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"


# Connect to MySQL database using SQLAlchemy's create_engine
engine = create_engine(mysql_uri)

# Execute SQL command
with engine.connect() as connection:
    sql_query = text('SELECT * FROM patients LIMIT 10')
    result = connection.execute(sql_query)
    # for row in result:
    #     print(row)


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

os.environ['OPENAI_API_KEY'] = "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"


# db = SQLDatabase.from_uri("sqlite:///./Chinook.db")
# toolkit = SQLDatabaseToolkit(db=db)

# agent_executor = create_sql_agent(
#     llm=OpenAI(temperature=0),
#     toolkit=toolkit,
#     verbose=True
# )

db = SQLDatabase.from_uri(mysql_uri)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
#toolkit = SQLDatabaseToolkit(db=db)
llm = OpenAI(temperature=0)
agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)
agent_executor.run("Describe the patient table")



