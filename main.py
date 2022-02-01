import config
import pyodbc
import sys
import pandas as pd
import numpy as np
from utils import process_text, process_text1, train_model, predict_text

if __name__ == '__main__':

    # Define database connector
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + config._SERVER +
                                                                ';DATABASE=' + config._DATABASE +
                                                                ';UID=' + config._USERNAME +
                                                                ';PWD=' + config._PASSWORD)
    
    if sys.argv[1] == 'Train':
        # Read data from database
        query = f"SELECT ObjetoProceso, Clasificacion FROM {config._SCHEMA}.{config._TRAINING_TABLE}"
        df = pd.read_sql(query,cnxn)
        
        # Process text: cleaning and word correction
        text, y = process_text(df)

        # Train model with updated data and save it 
        train_model(text, y)
            
    elif sys.argv[1] == 'Predict':
        # query = f"SELECT ObjetoProceso, Clasificacion FROM {config._SCHEMA}.{config._OPT_TABLE}"
        query = f"SELECT DetalleObjetoAContratar, Clasificacion FROM {config._SCHEMA}.{config._OPT_TABLE}"

        df = pd.read_sql(query,cnxn)

        # Process text: cleaning and word correction
        text, y = process_text1(df)
        print(y)
    
        pred = predict_text(text)
        print(pred)
        # set the new state in the table _OPT_TABLE acording the predictions
        # cursor = cnxn.cursor()
        # for t in pred:
        #     cursor.execute(f"INSERT INTO {config._SCHEMA}.{config._OPT_TABLE}(Estado) VALUES (?)", t)
        # cnxn.commit()