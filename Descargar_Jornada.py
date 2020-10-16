
from bs4 import BeautifulSoup, SoupStrainer
import requests
import lxml.html as lh
import write
import math
import pandas as pd
import csv


def exporto_datos(Datos,DF):
	
	for Datos_equipo in Datos:
		if len(Datos)>0:

			output=[]
			id=[]
			season=[]
			division=[]
			round=[]
			localTeam=[]
			visitorTeam=[]
			localGoals=[]
			visitorGoals=[]
			date=[]
			timestamp=[]
			
			for i,dato in enumerate(Datos_equipo):
				id.append(1)
				season.append(Datos_equipo[0][0][-4:] + '-' + str(int(Datos_equipo[0][0][-2:])+1))
				division.append(1)
				round.append(dato[2].split(' ')[1])
				localTeam.append(dato[3])
				visitorTeam.append(dato[7])
				localGoals.append(dato[5].split('-')[0].replace(' ',''))
				visitorGoals.append(dato[5].split('-')[1].replace(' ',''))
				date.append(dato[0])
				timestamp.append('000000')
			output=[id,season,division,round,localTeam,visitorTeam,localGoals,visitorGoals,date,timestamp]

			Dic={}
			headers=['id','season','division','round','localTeam','visitorTeam','localGoals','visitorGoals','date','timestamp']
			for i,k in enumerate(headers):
				Dic[k]=output[i]
			
			#Adjunto datos a DF
			# print(Dic)
			df_temp=pd.DataFrame(Dic)
			# df_datos.append(df_temp, ignore_index = False)
			frames=[DF,df_temp]
			result=pd.concat(frames)
			DF=result

			
    #Exporto a CSV
	DF.to_csv('Datos/FMEL_Dataset.csv',sep=',',encoding='utf-8',index=False)

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
	# exporto_datos(datos,df_datos)
	# print('*'*10)
	return datos
	
	
#obtengo las url de todos mis equipos

urls=[]
Ruta_equipos='Datos/equipos.csv'
Ruta_Datos='Datos/FMEL_Dataset.csv'
Datos_Equipos=pd.read_csv(Ruta_equipos)
df_Equipos=pd.DataFrame(Datos_Equipos)
Datos=pd.read_csv(Ruta_Datos)
df_datos=pd.DataFrame(Datos)

#Genero Copia de seguridad
import datetime
year = datetime.date.today().year
month = datetime.date.today().month
day = datetime.date.today().day
# hour = datetime.datetime.strftime('%H:%M')
# print(datetime.datetime.now())
# nombre='_' + str(year) + str(month) + str(day) + ' ' + str(hour)
nombre='_' + str(datetime.datetime.now()).replace(':','-')
df_datos.to_csv('Datos/Backup/FMEL_Dataset' + nombre + '.csv',sep=',',encoding='utf-8',index=False)


#Borro del dataset los datos de este año


df_datos=df_datos[df_datos.season!='2019-20']

urls=df_Equipos['urlData'].tolist()
equipos=df_Equipos['Equipo'].tolist()
Temporada='t2019'

D=[]
for i,url in enumerate(urls):
	Encontrado=True
	# for Temporada in Temporadas[1:]:
	print(' '*100,end="\r")
	print('Actualizando:',equipos[i],end="\r")
	#Obtengo datos

	D.append(get_data(url))

#Escribo del dataframe a CSV
print(' '*100,end="\r")

exporto_datos(D,df_datos)
print('Datos actualizados con éxito')
