from openai import ChatCompletion
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

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = "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"

# Initialize the OpenAI language model
llm = ChatCompletion()

# Database connection parameters
db_user = "mediyoga"
db_password = "mediyoga2023"
db_host = "3.111.133.67"
db_name = "Medi_Yoga"

# Construct the MySQL connection URI
mysql_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

# Connect to the database
db = SQLDatabase.from_uri(mysql_uri)

# Function to parse query result into a list of elements
def query_as_list(db, query):
    res = db.run(query)
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))


examples = [
    {
        "question": "Who lived longer, Muhammad Ali or Alan Turing?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali
""",
    },
    {
        "question": "When was the founder of craigslist born?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952
""",
    },
    {
        "question": "Who was the maternal grandfather of George Washington?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball
""",
    },
    {
        "question": "Are both the directors of Jaws and Casino Royale from the same country?",
        "answer": """
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
So the final answer is: No
""",
    },
]


# Create vector store with all the distinct proper nouns from the database
artists = query_as_list(db, "SELECT first_name FROM patients")
albums = query_as_list(db, "SELECT appointment_date FROM appointment")

# Define system message template
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
    input_variables=["question", "answer"], template="Question: {question}\n{answer}"
)

#print(example_prompt.format(**examples[0]))

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
    prefix=system_prefix,
)


# Create few-shot prompt template
# few_shot_prompt = PromptTemplate(
#     examples=examples    
#     example_prompt=PromptTemplate.from_template(
#         "User input: {input}\nSQL query: {query}"
#     ),
#     input_variables=["input", "dialect", "top_k"],
#     prefix=system_prefix,
#     suffix="",
# )

print("test")
print(prompt.format(input="Who was the father of Mary Ball Washington?"))


# Create full prompt
# full_prompt = ChatPromptTemplate.from_messages(
#     [
#         SystemMessagePromptTemplate(prompt=few_shot_prompt),
#         ("human", "{input}"),
#         MessagesPlaceholder("agent_scratchpad"),
#     ]
# )

# Define function to create the SQL agent
# def create_sql_agent(llm, db, prompt, verbose=True, agent_type="openai-tools"):
#     # You can implement your agent creation logic here
#     pass

# # Create agent
# agent = create_sql_agent(
#     llm=llm,
#     db=db,
#     prompt=full_prompt,
#     agent_type="openai-tools",
#     verbose=True,
# )

# # Test the agent
# response = agent.invoke({"input": "How many appointments are scheduled for today?"})
# print(response)
