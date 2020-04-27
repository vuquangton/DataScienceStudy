import pyodbc
import pandas as pd

class MSSQLHelper:
    
    def __init__(self, driver,server, db,user, pwd):
        super().__init__()
    
    def read_data():    
        conn = pyodbc.connect("""
        Driver={ODBC Driver 17 for SQL Server};Server=.;Database=db_ML;Trusted_Connection=yes;
        ;""")
        df = pd.read_sql_query('select  TX_NO, SKU, QUAN from view_dm_sale_retails_151', conn)
        return df

