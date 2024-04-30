from openai import ChatCompletion
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import (
    FewShotPromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.utilities.sql_database import SQLDatabase

import os
import re
import getpass



os.environ['OPENAI_API_KEY'] = "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"
#os.environ["OPENAI_API_KEY"] = getpass.getpass()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


db_user = "mediyoga"
db_password = "mediyoga2023"
db_host = "3.111.133.67"
db_name = "Medi_Yoga"

mysql_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

db = SQLDatabase.from_uri(mysql_uri)

#db.run("SELECT * FROM patients LIMIT 10;")

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)


examples = [
    {"input": "List all artirst.", "query": "SELECT * FROM patients;"},
    {
        "input": "Find all albums for the artist 'AC/DC'.",
        "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
    },
    {
        "input": "List all tracks in the 'Rock' genre.",
        "query": "SELECT first_name FROM patients WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');",
    },
    {
        "input": "Find the total duration of all tracks.",
        "query": "SELECT SUM(Milliseconds) FROM Track;",
    },
    {
        "input": "List all customers from Canada.",
        "query": "SELECT * FROM Customer WHERE Country = 'Canada';",
    },
    {
        "input": "How many tracks are there in the album with ID 5?",
        "query": "SELECT COUNT(*) FROM Track WHERE AlbumId = 5;",
    },
    {
        "input": "Find the total number of invoices.",
        "query": "SELECT COUNT(*) FROM Invoice;",
    },
    {
        "input": "List all tracks that are longer than 5 minutes.",
        "query": "SELECT * FROM Track WHERE Milliseconds > 300000;",
    },
    {
        "input": "Who are the top 5 customers by total purchase?",
        "query": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
    },
    {
        "input": "Which albums are from the year 2000?",
        "query": "SELECT * FROM Album WHERE strftime('%Y', ReleaseDate) = '2000';",
    },
    {
        "input": "How many employees are there",
        "query": 'SELECT COUNT(*) FROM "Employee"',
    },
]

system_prefix = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

Here are some examples of user inputs and their corresponding SQL queries:"""

example_prompt = PromptTemplate(
    input_variables=["input", "query"], template="Question: {input}\n{query}"
)

#print(example_prompt.format(**examples[0]))

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input","dialect","top_k"],
    prefix=system_prefix,
)
print("test")

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)


prompt_val = full_prompt.invoke(
    {
        "input": "How many artists are there?",
        "top_k": 5,
        "dialect": "SQLite",
        "agent_scratchpad": [],
    }
)

#print(prompt_val.to_string())


agent = create_sql_agent(
    llm=llm,
    db=db,
    prompt=full_prompt,
    verbose=True,
    agent_type="openai-tools",
)

agent.invoke({"input": "How many patients are there?"})

#print(prompt.format(input="Who was the father of Mary Ball Washington?"))

#print(prompt.format(input="How many arists are there?", dialect="SQL", top_k=5))


