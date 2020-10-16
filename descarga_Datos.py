
from bs4 import BeautifulSoup, SoupStrainer
import requests
import lxml.html as lh
import write
import Apuestas_3
import math
import pandas as pd

def exporto_datos(Datos):
    if len(Datos)>0:
        headers=['id','season','division','round','localTeam','visitorTeam','localGoals','visitorGoals','date','timestamp']
        season=Datos[0][0][-4:] + '-' + str(int(Datos[0][0][-2:])+1)
        division=1
        id='1'
        timestamp='000000'
        output=[]
        for dato in Datos:
            
            round=dato[2].split(' ')[1]
            localTeam=dato[3]
            visitorTeam=dato[7]
            localGoals=dato[5].split('-')[0].replace(' ','')
            visitorGoals=dato[5].split('-')[1].replace(' ','')
            date=dato[0]
            output.append([id,season,division,round,localTeam,visitorTeam,localGoals,visitorGoals,date,timestamp])
        write.write_file('descargas','datos',headers,output,'a')
        

def get_data(url):
	
	#Create a handle, page, to handle the contents of the website
	page = requests.get(url)
	#Store the contents of the website under doc
	doc = lh.fromstring(page.content)
	#Parse data that are stored between <tr>..</tr> of HTML
	tr_elements = doc.xpath('//tr')
	# print(tr_elements)

	tr_elements = doc.xpath('//tr')
	#Create empty list
	col=[]
	i=0
	#For each row, store each first element (header) and an empty list
	inicio=False
	datos=[]
	for i,x in enumerate(tr_elements):
		if len(x)>0:
			# print(x[0].text_content)
			if x[0].text_content()=='Fecha':
				inicio=True
		if inicio==True and len(x)==8:
			fila=[]
			for t in x:
				if x[1].text_content()=='Liga':
					fila.append(t.text_content())
			if len(fila)>0:
				datos.append(fila)
	exporto_datos(datos)
	# print('*'*10)
	return datos
	
	
#obtengo las url de todos mis equipos

urls=[]
Ruta_equipos='Datos/equipos.csv'
Datos_Equipos=pd.read_csv(Ruta_equipos)
df_Equipos=pd.DataFrame(Datos_Equipos)

urls=df_Equipos['urlData'].tolist()
equipos=df_Equipos['Equipo'].tolist()
# Temporadas=['t2019','t2018','t2017']
Temporadas=[]
Temporadas.append('t2019')
for t in range(42,70):
    Temporadas.append('t19'+str(t))


for i,url in enumerate(urls):
    Encontrado=True
	# for Temporada in Temporadas[1:]:
    for Temporada in Temporadas[1:]:
        
        print('Equipo:',equipos[i])
        #Obtengo datos
        if Encontrado==True:
            D=[]
            D.append(get_data(url))
            page = requests.get(url)    
            data = page.text
            soup = BeautifulSoup(data,"lxml")
        Encontrado=False
        for link in soup.find_all('a'):
            if link.get('href') is not None:
                if link.get('href')[:5]==Temporada:
                    print(Temporada)
                    url="https://www.bdfutbol.com/es/t/" + link.get('href') + "?tab=partits"
                    Encontrado=True
                    break
            Encontrado=False
                    