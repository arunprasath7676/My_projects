import os
from langchain.llms.openai import OpenAI
from langchain.sql_database import SQLDatabase
from lcforecast.agentkit import ForecastToolkit
from lcforecast.agentkit.base import create_forecast_agent


os.environ['OPENAI_API_KEY'] = "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"


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


db_user = "mediyoga"
db_password = "mediyoga2023"
db_host = "3.111.133.67"
db_name = "Medi_Yoga"

mysql_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

db = SQLDatabase.from_uri(mysql_uri)

#llm = OpenAI(temperature=0)

#db = SQLDatabase.from_uri(os.environ["DB_CONN"])

#mysql_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"


db = SQLDatabase.from_uri(mysql_uri)


#toolkit = SQLDatabaseToolkit(db=db, llm=llm)

#agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

#agent_executor.run("show me all the appointments today?")

toolkit = ForecastToolkit(db=db, llm=llm)

agent_executor = create_forecast_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

agent_executor.run("Forecast the invoice total for next month based on the last 3 months of appointment")