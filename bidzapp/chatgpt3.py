import os
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain.agents import create_sql_agent, AgentExecutor

os.environ['OPENAI_API_KEY'] = "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"


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


db_user = "mediyoga"
db_password = "mediyoga2023"
db_host = "3.111.133.67"
db_name = "Medi_Yoga"

mysql_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

db = SQLDatabase.from_uri(mysql_uri)

schema = """
-appointment(token,appointment_date,appointment_slot,appointment_information,appointment_status,
is_first,appointment_time,case_category_token,check_in_notification_triggered,check_in_time,
check_out_time,patients_token,room,severity,marked_fc,has_reports,doctor_details_token,
notes,is_flagged,deleted_at,created_at,updated_at)
-patient(token,patient_id,image,prefix,first_name,last_name,country_code,mobile_number,alt_country_code,
alternate_mobile_number,email,dob,gender,address,allergy,health,user_status,app_status,
users_token,is_primary,on_going_case_id,area,city,state)
"""

#tables = ["appointment"]

tables = "appointment"



missing_tables = [table for table in tables if table not in schema]

if missing_tables:
    print(f"Error: The following tables not available in the schema")
else:


    
   # toolkit = SQLDatabaseToolkit(db=db, schema=schema, tables=tables, llm=llm)

    #print("printing toolkit", toolkit)
   
    #agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

   # agent_executor.run("what is the price of vegetables")

   # Initialize SQLDatabaseToolkit with the specified table
    toolkit = SQLDatabaseToolkit(db=db, tables=[tables], llm=llm)

    # Create SQL agent
    agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

    # Run the query
    agent_executor.run("how many appointments today?")
