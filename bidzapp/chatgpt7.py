from openai import ChatCompletion
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import (
    FewShotPromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain.agents.agent_toolkits import create_retriever_tool
#from langchain_openai import OpenAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from openai import ChatCompletion
import ast
import os
import re


os.environ['OPENAI_API_KEY'] = "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"


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


db_user = "mediyoga"
db_password = "mediyoga2023"
db_host = "3.111.133.67"
db_name = "Medi_Yoga"

mysql_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

db = SQLDatabase.from_uri(mysql_uri)

def query_as_list(db, query):
    res = db.run(query)
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))


artists = query_as_list(db, "SELECT first_name FROM patients")
albums = query_as_list(db, "SELECT appointment_date FROM appointment")
albums[:5]


vector_db = FAISS.from_texts(artists + albums, OpenAIEmbeddings())
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
valid proper nouns. Use the noun most similar to the search."""
retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)

system = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool! 

You have access to the following tables: {table_names}

If the question does not seem related to the database, just return "I don't know" as the answer."""

prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{input}"), MessagesPlaceholder("agent_scratchpad")]
)
agent = create_sql_agent(
    llm=llm,
    db=db,
    extra_tools=[retriever_tool],
    prompt=prompt,
    agent_type="openai-tools",
    verbose=True,
)


agent.invoke({"input": "who paid the highest payment?"})
