import requests
import lxml.html as lh
import pandas as pd



def get_data(url):
	
	#Create a handle, page, to handle the contents of the website
	page = requests.get(url)
	#Store the contents of the website under doc
	doc = lh.fromstring(page.content)
	#Parse data that are stored between <tr>..</tr> of HTML
	tr_elements = doc.xpath('//tr')
	# print(tr_elements)

	#Check the length of the first 12 rows
	# [len(T) for T in tr_elements[:15]]

	tr_elements = doc.xpath('//tr')
	#Create empty list
	col=[]
	i=0
	#For each row, store each first element (header) and an empty list
	inicio=False
	datos=[]
	for i,x in enumerate(tr_elements):
		if len(x)>0:
			# print(x[0].text_content())
			if x[0].text_content()=='Fecha':
				# print('efectivamente se ha iniciado')
				inicio=True
		if inicio==True and (len(x)==9 or len(x)==8):

			datos.append([t.text_content() for t in x])

	return datos

def estadisticas(Equipo,url,pos):

	Total_Partidos_Local=0
	Total_Partidos_Visitante=0
	Total_Partidos_Ganados_Local=0
	Total_Partidos_Ganados_Visitante=0
	Total_Goles_Local=0
	Total_Goles_Visitante=0
	datos=get_data(url)

	for l in datos:
		if l[1]=='Liga':
			# print(l[3],Equipo)
			if l[3]==Equipo:
				Total_Partidos_Local+=1
				Total_Goles_Local+=int(l[5].split('-')[0])
				if int(l[5].split('-')[0])>int(l[5].split('-')[1]):
					Total_Partidos_Ganados_Local+=1
			else:
				Total_Partidos_Visitante+=1
				Total_Goles_Visitante+=int(l[5].split('-')[1])
				if int(l[5].split('-')[0])<int(l[5].split('-')[1]):
					Total_Partidos_Ganados_Visitante+=1
			
		# print(l)
	if pos==0: #Local

		
		# print('Total partidos como local:',Total_Partidos_Local,'Goles:',Total_Goles_Local,'ratio ganados local',Total_Partidos_Ganados_Local/Total_Partidos_Local)
		return(Total_Partidos_Ganados_Local/Total_Partidos_Local,Total_Goles_Local/Total_Partidos_Local)
	elif pos==1:
		# print('Total partidos como visitante:',Total_Partidos_Visitante,'Goles:',Total_Goles_Visitante,'ratio ganados visitante',Total_Partidos_Ganados_Visitante/Total_Partidos_Visitante)
		return(Total_Partidos_Ganados_Visitante/Total_Partidos_Visitante,Total_Goles_Visitante/Total_Partidos_Visitante)


	