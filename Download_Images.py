#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Description:   Programm to download pictures to folder
#Authors:       Kevin Jordi, Sandro Bürgler, This Steinmetz
#Version:       1.0
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Import Modules
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import urllib.request
import pandas as pd 
import numpy as np
import urllib
import math
import uuid


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Variables
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DATA = 'zuerich-wie-neu.csv'

#read DATA from CSV
csv = pd.read_csv(DATA)
data = (csv[['service_name','media_url']])
#urls = (csv['media_url'])

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Code
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#delete rows with empty fields
data.dropna(inplace=True)

#Clean duplicated url entries
data.drop_duplicates(['media_url'])
                  
#save pictures in folders, one folder for each category

for  index, row in data.iterrows():
  
    if row['service_name'] == 'Abfall/Sammelstelle':
        try:
            print(row['media_url'])
            filename = str(uuid.uuid4())
            urllib.request.urlretrieve(row['media_url'], "Images/Abfall/"+filename+".jpeg")
        except:
            print("URL:"+ row['media_url']+ " not found") 
    elif row['service_name'] == 'Allgemein':
        try: 
            print(row['media_url'])
            filename = str(uuid.uuid4())
            urllib.request.urlretrieve(row['media_url'], "Images/Allgemein/"+filename+".jpeg")
        except:
            print("URL:"+ row['media_url']+ " not found") 
    elif row['service_name'] == 'Beleuchtung/Uhren':
        try:
            print(row['media_url'])
            filename = str(uuid.uuid4())
            urllib.request.urlretrieve(row['media_url'], "Images/Beleuchtung/"+filename+".jpeg")
        except:
            print("URL:"+ row['media_url']+ " not found")    
    elif row['service_name'] == 'Brunnen/Hydranten':
        try:
            print(row['media_url'])
            filename = str(uuid.uuid4())
            urllib.request.urlretrieve(row['media_url'], "Images/Brunnen/"+filename+".jpeg")
        except:
            print("URL:"+ row['media_url']+ " not found") 
    elif row['service_name'] == 'Graffiti':
        try:
            print(row['media_url'])
            filename = str(uuid.uuid4())
            urllib.request.urlretrieve(row['media_url'], "Images/Grafitti/"+filename+".jpeg")
        except:
            print("URL:"+ row['media_url']+ " not found")              
    elif row['service_name'] == 'Grünflächen/Spielplätze':
        try:    
            print(row['media_url'])
            filename = str(uuid.uuid4())
            urllib.request.urlretrieve(row['media_url'], "Images/Gruenflaechen/"+filename+".jpeg")
        except:
            print("URL:"+ row['media_url']+ " not found")
    elif row['service_name'] == 'Schädlinge':
        try:    
            print(row['media_url'])
            filename = str(uuid.uuid4())
            urllib.request.urlretrieve(row['media_url'], "Images/Schaedlinge/"+filename+".jpeg")
        except:
            print("URL:"+ row['media_url']+ " not found")  
    elif row['service_name'] == 'Signalisation/Lichtsignal':
        try:    
            print(row['media_url'])
            filename = str(uuid.uuid4())
            urllib.request.urlretrieve(row['media_url'], "Images/Signalisation/"+filename+".jpeg")
        except:
            print("URL:"+ row['media_url']+ " not found")     
    elif row['service_name'] == 'Strasse/Trottoir/Platz':
        try:
            print(row['media_url'])
            filename = str(uuid.uuid4())
            urllib.request.urlretrieve(row['media_url'], "Images/Strasse/"+filename+".jpeg")   
        except:
            print("URL:"+ row['media_url']+ " not found") 
    elif row['service_name'] == 'VBZ/ÖV':
        try:
            print(row['media_url'])
            filename = str(uuid.uuid4())
            urllib.request.urlretrieve(row['media_url'], "Images/VBZ/"+filename+".jpeg")
        except:
            print("URL:"+ row['media_url']+ " not found")   
                                             
