#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Description:   Programm to extract the description to text files
#Authors:       Kevin Jordi, Sandro Bürgler, This Steinmetz
#Version:       1.0
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Import Modules
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pathlib
import pandas as pd 
import numpy as np
import uuid

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#functions
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def printerror():
        print("Wrong encoding or something else wrong with entry ")
        print(i)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Attributes
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DATA = 'zuerich-wie-neu.csv'

#read DATA from CSV
csv = pd.read_csv(DATA)
data = (csv[['service_name','description']])
print(data)
i = 0

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Code
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#delete rows with empty fields
data.dropna(inplace=True)

#Clean duplicated descriptions
data.drop_duplicates(['description'])
                  
#save pictures in folders, one folder for each category

for  index, row in data.iterrows():
        i = i +1
        filename = str(uuid.uuid4())
        if row['service_name'] == 'Abfall/Sammelstelle':
                try:
                        value = row['description']
                        pathlib.Path("Text/Abfall/"+filename+".txt").write_text(f"{value}")
                except:
                        printerror()
        elif row['service_name'] == 'Allgemein':
                try:
                        value = row['description']
                        pathlib.Path("Text/Allgemein/"+filename+".txt").write_text(f"{value}")
                except:
                        printerror()
        elif row['service_name'] == 'Beleuchtung/Uhren':
                try:
                        value = row['description']
                        pathlib.Path("Text/Beleuchtung/"+filename+".txt").write_text(f"{value}")
                except:
                        printerror()
        elif row['service_name'] == 'Brunnen/Hydranten':
                try:
                        value = row['description']
                        pathlib.Path("Text/Brunnen/"+filename+".txt").write_text(f"{value}")
                except:
                        printerror()
        elif row['service_name'] == 'Graffiti':
                try:
                        value = row['description']
                        pathlib.Path("Text/Graffiti/"+filename+".txt").write_text(f"{value}")
                except:
                        printerror()        
        elif row['service_name'] == 'Grünflächen/Spielplätze':
                try:
                        value = row['description']
                        pathlib.Path("Text/Gruenflaeche/"+filename+".txt").write_text(f"{value}")
                except:
                        printerror()
        elif row['service_name'] == 'Schädlinge':
                try:
                        value = row['description']
                        pathlib.Path("Text/Schaedlinge/"+filename+".txt").write_text(f"{value}")
                except:
                        printerror()
        elif row['service_name'] == 'Signalisation/Lichtsignal':
                try:
                        value = row['description']
                        pathlib.Path("Text/Signalisation/"+filename+".txt").write_text(f"{value}")
                except:
                        printerror()
        elif row['service_name'] == 'Strasse/Trottoir/Platz':
                try:
                        value = row['description']
                        pathlib.Path("Text/Strasse/"+filename+".txt").write_text(f"{value}")
                except:
                        printerror()  
        elif row['service_name'] == 'VBZ/ÖV':
                try:
                        value = row['description']
                        pathlib.Path("Text/VBZ/"+filename+".txt").write_text(f"{value}")
                except:
                        printerror()
     

                                             
