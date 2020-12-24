import os, numpy, PIL, cv2
from PIL import Image
from tqdm import tqdm

DATADIR = "C:/bereinigt"
CATEGORIES = ["Abfall", "Beleuchtung","Brunnen", "Graffiti", "Gruenflaechen", "Schaedlinge", "Signalisation", "Strasse", "VBZ"]
hoehe = 0
breite = 0
divisor = 0


for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  # create path to categories
        class_num = CATEGORIES.index(category)  # get the classification 

        for img in tqdm(os.listdir(path)):  # iterate over each image 
            try:
                
                bild =  cv2.imread(os.path.join(path,img))
                h, w, _ = bild.shape
                breite = breite + w
                hoehe = hoehe + h
                divisor = divisor + 1
                
            except Exception as e:  # in the interest in keeping the output clean...
                pass

print("Total pixel Breite: ", breite)
print("Total pixel Höhe: ", hoehe)
print("Anzahl Bilder: ", divisor)

avbreite = breite/ divisor
avhoehe = hoehe /divisor

print("Durchschnittliche pixel Breite: ", (int(avbreite)))
print("Durchschnittliche pixel Höhe: ", (int(avhoehe)))