import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy import array as ARR
from numpy import random as R
from numpy import asanyarray as asr

import pylab as pl
from sklearn import preprocessing
from sklearn import linear_model
import time
import requests
import lxml.html as lh

class Apuestas:

	Ruta_equipos='Datos/equipos.csv'
	Ruta_Datos='Datos/FMEL_Dataset.csv'
	Equipos=''
	Datos=''
	Model_Data=''
	Datos_Corregidos=''
	Encuentro_posible=False
	Resultados_predecidos={}
	Advertencia=''
	Equipo_Local={'Nombre':'',
				  'id':0,
				  'url':'',
				'Total_Partidos_Local':0,
				'Total_Partidos_Ganados_Local':0,
				'Total_Goles_Local':0,

				}

	Equipo_Visitante={'Nombre':'',
				'id':0,
				'url':'',
			'Total_Partidos_Visitante':0,
			'Total_Partidos_Ganados_Visitante':0,
			'Total_Goles_Visitante':0,

			}

	def get_data_equipo_web(self,url):
	
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

	def Estadisticas_Encuentro(self):

		for i in range(0,2):


			Total_Partidos_Local=0
			Total_Partidos_Visitante=0
			Total_Partidos_Ganados_Local=0
			Total_Partidos_Ganados_Visitante=0
			Total_Goles_Local=0
			Total_Goles_Visitante=0

			if i==0:
				url=self.Equipo_Local['url']
				Equipo=self.Equipo_Local['Nombre']
			else:
				url=self.Equipo_Visitante['url']
				Equipo=self.Equipo_Visitante['Nombre']

			datos=self.get_data_equipo_web(url)

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
			if i==0: #Local

				self.Equipo_Local['Total_Partidos_Local']=Total_Partidos_Local
				self.Equipo_Local['Total_Partidos_Ganados_Local']=Total_Partidos_Ganados_Local
				self.Equipo_Local['Total_Goles_Local']=Total_Partidos_Ganados_Local
				# print('Total partidos como local:',Total_Partidos_Local,'Goles:',Total_Goles_Local,'ratio ganados local',Total_Partidos_Ganados_Local/Total_Partidos_Local)
			else:

				self.Equipo_Visitante['Total_Partidos_Visitante']=Total_Partidos_Visitante
				self.Equipo_Visitante['Total_Partidos_Ganados_Visitante']=Total_Partidos_Ganados_Visitante
				self.Equipo_Visitante['Total_Goles_Visitante']=Total_Goles_Visitante
				# print('Total partidos como visitante:',Total_Partidos_Visitante,'Goles:',Total_Goles_Visitante,'ratio ganados visitante',Total_Partidos_Ganados_Visitante/Total_Partidos_Visitante)
				# return(Total_Partidos_Ganados_Visitante/Total_Partidos_Visitante,Total_Goles_Visitante/Total_Partidos_Visitante)


	def set_encuentro(self,Local,Visitante):

		#Fijo los valores correspondientes a los equipos que se van a encontrar

		self.Equipo_Local['Nombre']=Local
		self.Equipo_Visitante['Nombre']=Visitante		
		self.Equipo_Local['id']=self.Equipos.loc[self.Equipos.Equipo==Local]['id'].values[0]
		self.Equipo_Visitante['id']=self.Equipos.loc[self.Equipos.Equipo==Visitante]['id'].values[0]
		self.Equipo_Local['url']=self.Equipos.loc[self.Equipos.Equipo==Local]['urlData'].values[0]
		self.Equipo_Visitante['url']=self.Equipos.loc[self.Equipos.Equipo==Visitante]['urlData'].values[0]
		

		self.check_encuentro()


	def gen_intervalos(self,Diccionario):

		Minimo=min(Diccionario.keys())
		Maximo=max(Diccionario.keys())

		if Minimo<1:
			Intervalos=[[0,0.5,0,0]]
			Minimo+=0.5
		else:Intervalos=[]
		for i in range(int(Minimo),int(round(Maximo+1,0))):
			Intervalos.append([i-0.5,i+0.5,0,i])
			


		#Genero intervalos agrupados

		for key in Diccionario.keys():
			for inter in Intervalos:
				if key>=inter[0] and key<=inter[1]:
					inter[2]+=Diccionario[key]
					break
		
		return Intervalos

	def CorrijoDatos(self):

		df=self.Datos
		# Quito columnas inutiles
		df.drop('round', axis='columns',inplace=True)
		df.drop('date', axis='columns',inplace=True)
		df.drop('timestamp', axis='columns',inplace=True)
		df.drop('division', axis='columns',inplace=True)
		
		#Creo una columna Año
		df['año']=df['season'].str[:4].astype('int32')
		
		GolesLCorregidos=[]
		GolesVCorregidos=[]
		for i,datos in enumerate(df.iterrows()):

			if datos[1][6]<1970:
				golesL= ((1970-datos[1][6])*(-0.0591))+(datos[1][4])
				golesV= ((1970-datos[1][6])*(-0.0591))+(datos[1][5])
				if golesL<0:
					golesL=0
				if golesV<0:
					golesV=0
				
				GolesLCorregidos.append(round(golesL,2))
				GolesVCorregidos.append(round(golesV,2))
				
				# print('Año',datos[1][6],'Tgoles:',datos[1][4]+datos[1][5],'corregido',goles)
			else:
				GolesLCorregidos.append(datos[1][4])
				GolesVCorregidos.append(datos[1][5])
		
			
		
		df['visitorGoals']=GolesVCorregidos
		df['localGoals']=GolesLCorregidos

		return df

	def Grafico_dist_goles(TGoles,Estadisticas,Local,Visitante,TotalGoles_prediccion,acumulado,acumulado2,Warning):

		hist_goles=pd.DataFrame()
		hist_goles['Pgoles']=(TGoles.groupby(Estadisticas.TotalGoles).agg('count'))/TGoles.groupby(Estadisticas.TotalGoles).agg('count').sum()

		
		# plt.figure(1)
		plt.figure(figsize=(18.0, 12.0))
		plt.subplot(222)
		
		if max(hist_goles.index)>max(TotalGoles_prediccion.keys()):
			M=max(hist_goles.index)+0.5
		else:M=max(TotalGoles_prediccion.keys())+0.5
		
		acum=0
		x=[]
		y1=[]
		y2=[]
		for index, row in hist_goles.iterrows():
			acum+=row['Pgoles']*100
			x.append(index)
			y1.append(row['Pgoles']*100)
			y2.append(acum)
			# print(index)
			if index-int(index)==0:
				plt.text(index-0.1,acum+2,str(round(acum,1))+'%')
				plt.text(index,row['Pgoles']*100-2,str(round(row['Pgoles']*100,1))+'%')
				
		plt.grid()
		plt.title('Distribución de goles histórica entre: ' +Local +' vs ' + Visitante)
		plt.ylabel('Probabilidad')
		plt.xlabel('# de goles totales')		
		plt.plot(x,y1,color='skyblue',label='Prob')
		plt.plot(x,y2, color='lightpink',label='Prob Acumulada')
		plt.fill_between(x, y2,color="lightpink", alpha=0.4)
		plt.fill_between(x, y1,color="skyblue", alpha=0.6)
		plt.legend(loc='best',shadow=True,fontsize='small')
		plt.ylim((0, 110))
		plt.xlim((-0.5, M))
		
		plt.subplot(221)# Grafico 2
		plt.text(0,102,Warning[0])
		x1=[]
		y3=[]
		
		i=0
		for valor in sorted(TotalGoles_prediccion.keys()):
			# print(valor)
			x1.append(valor)
			y3.append(TotalGoles_prediccion[valor]*100 / sum(TotalGoles_prediccion.values()))
			# plt.text(valor,TotalGoles_prediccion[valor]*100 / sum(TotalGoles_prediccion.values()),TotalGoles_prediccion[valor]*100 / sum(TotalGoles_prediccion.values()))

		for i,intervalo in enumerate(acumulado):
			# plt.bar(intervalo[3], intervalo[2],color=(0.8, 0.4, 0.4, 0.7), width=0.8, bottom=None, align='center')
			plt.bar(intervalo[3], intervalo[2],color="lightpink", width=0.8, bottom=None, align='center', alpha=0.4)
			plt.text(intervalo[3],intervalo[2],str(intervalo[2]) + '%')

		
		plt.title('Predicción de la distribución de goles totales')
		plt.xlabel('# de goles totales')
		plt.ylabel('Probabilidad')
		plt.grid()
		plt.plot(x1,y3, color='gold',label='Distribución de goles')
		plt.legend(loc='best',shadow=True,fontsize='small')
		plt.fill_between(x1, y3,color="gold", alpha=0.4)
		plt.ylim((0, 110))
		plt.xlim((-0.5, M))
		
		plt.subplot(223)# Grafico 3
		
		total_partidos=0
		for partido in acumulado2:
			total_partidos+=partido[2]
		
		for i,intervalo in enumerate(acumulado2):
			# plt.bar(intervalo[3], intervalo[2],color=(0.8, 0.4, 0.4, 0.7), width=0.8, bottom=None, align='center')
			y=round(intervalo[2]*100/total_partidos,0)
			plt.bar(intervalo[3], y,color="lightpink", width=0.8, bottom=None, align='center', alpha=0.4)
			plt.text(intervalo[3],y,str(y) + '%')

		
		plt.title('Predicción de partidos similares KNN')
		plt.xlabel('# de goles totales')
		plt.ylabel('Probabilidad')
		plt.grid()
		plt.ylim((0, 110))
		plt.xlim((-0.5, M))
		plt.text(0,102,Warning[1])
		plt.subplot(224)# Grafico 3
		
		Final={}
		for i in range(int(M)+1):
			Final[i]=[0,0] #key=goles [%,repeticiones]
			
			for ele in acumulado:
			
				if ele[3]==i:
					porcentaje=Final[i][0]+ele[2]
					rep=Final[i][1]+1
					Final[i]=[porcentaje,rep]
			
			for ele in acumulado2:
			
				if ele[3]==i:
					porcentaje=Final[i][0]+ele[2]*100/total_partidos
					rep=Final[i][1]+1
					Final[i]=[porcentaje,rep]
			
			
		
		for key in Final:
			try:
				y=round(Final[key][0]/Final[key][1],1)
				plt.bar(key,y ,color="lightpink", width=0.8, bottom=None, align='center', alpha=0.4)
				plt.text(key,y,str(y) + '%')
			except:pass

		
		plt.title('Predicción media de los modelos')
		plt.xlabel('# de goles totales')
		plt.ylabel('Probabilidad')
		plt.grid()
		plt.ylim((0, 110))
		plt.xlim((-0.5, M))
		
		t = time.localtime()
		timestamp = time.strftime('%Y-%b-%d_%H%M', t)
		
		
		plt.savefig('predicciones/'+Local+'-'+Visitante+' ' + str(timestamp)+'.png',dpi=300)
		plt.show()

	def make_noise(self):

		n=100
		var=0.075
		Ganados_L=R.normal(self.Equipo_Local['Total_Partidos_Ganados_Local']/self.Equipo_Local['Total_Partidos_Local'], var, n).tolist()
		Ganados_V=R.normal(self.Equipo_Visitante['Total_Partidos_Ganados_Visitante']/self.Equipo_Visitante['Total_Partidos_Visitante'], var, n).tolist()
		Goles_L=R.normal(self.Equipo_Local['Total_Goles_Local']/self.Equipo_Local['Total_Partidos_Local'], var, n).tolist()
		Goles_V=R.normal(self.Equipo_Visitante['Total_Goles_Visitante']/self.Equipo_Visitante['Total_Partidos_Visitante'], var, n).tolist()

		ruido = list(zip(Ganados_L,Ganados_V,Goles_L,Goles_V))
		return ruido

	def RegresionLineal(self,datos):

		#Creo el modelo lineal 

		df=self.Model_Data
		
		regr = linear_model.LinearRegression()
		# x = np.asanyarray(train[['Ganados_L','Empatados_L','Perdidos_L','Ganados_V','Empatados_V','Perdidos_V','Goles_L','Goles_V']])
		x = asr(df[['Ganados_L','Ganados_V','Goles_L','Goles_V']])
		y = asr(df[['TotalGoles']])
		regr.fit (x, y)
		# The coefficients
		# print ('Coefficients: ', regr.coef_)

		#compruebo resultados
		TotalGoles=regr.predict([datos])
		# y_hat= regr.predict(test[['Ganados_L','Ganados_V','Goles_L','Goles_V']])
		return TotalGoles

	def get_equipos(self):

		Datos_Equipos=pd.read_csv(self.Ruta_equipos)
		return Datos_Equipos.sort_values(by=['Equipo'])

	def get_data(self):

		Datos=pd.read_csv(self.Ruta_Datos)

		return Datos

	def check_encuentro(self):
		if self.Equipo_Local['Nombre'] != self.Equipo_Visitante['Nombre']: self.Encuentro_posible=True
		else: self.Encuentro_posible=False

	def DatosModelo(self):

		Local=self.Equipo_Local['Nombre']
		Visitante=self.Equipo_Visitante['Nombre']
		
		df=self.CorrijoDatos()

		Datos_Encuentros=df[(df.localTeam==Local) & (df.visitorTeam==Visitante)]
		Goles_Encuentros=(Datos_Encuentros['visitorGoals'].groupby(Datos_Encuentros.season).agg('sum'))+(Datos_Encuentros['localGoals'].groupby(Datos_Encuentros.season).agg('sum'))

		#obtengo partidos jugados como local

		Jugados_Local=df[df.localTeam==Local]
		JL=(Jugados_Local['season'].groupby(Jugados_Local.season).agg('count'))


		#obtengo el ratio de partidos ganados en casa por temporada

		Ganados_local=df[(df.localGoals>df.visitorGoals) & (df.localTeam==Local)]
		GL=(Ganados_local['season'].groupby(Ganados_local.season).agg('count'))

		#obtengo el ratio de partidos empatados en casa por temporada
		Empatados_Local=df[(df.localGoals==df.visitorGoals) & (df.localTeam==Local)]
		EL=(Empatados_Local['season'].groupby(Empatados_Local.season).agg('count'))

		#obtengo el ratio de partidos perdidos en casa por temporada
		Perdidos_Local=df[(df.localGoals<df.visitorGoals) & (df.localTeam==Local)]
		PL=(Perdidos_Local['season'].groupby(Perdidos_Local.season).agg('count'))

		#Obtengo el n de goles marcado como local
		Goles_local=df[df.localTeam==Local]
		Goles_L=(Goles_local['localGoals'].groupby(Goles_local.season).agg('sum'))
		# print(Goles_L)

		#obtengo partidos jugados como Visitante

		Jugados_Visitante=df[df.visitorTeam==Visitante]
		JV=(Jugados_Visitante['season'].groupby(Jugados_Visitante.season).agg('count'))

		#Obtengo el ratio de partidos ganados fuera casa por temporada

		Ganados_Visitante=df[(df.localGoals<df.visitorGoals) & (df.visitorTeam==Visitante)]
		GV=(Ganados_Visitante['season'].groupby(Ganados_Visitante.season).agg('count'))

		#obtengo el ratio de partidos empatados en casa por temporada
		Empatados_Visitante=df[(df.localGoals==df.visitorGoals) & (df.visitorTeam==Visitante)]
		EV=(Empatados_Visitante['season'].groupby(Empatados_Visitante.season).agg('count'))

		#obtengo el ratio de partidos perdidos en casa por temporada
		Perdidos_Visitante=df[(df.localGoals>df.visitorGoals) & (df.visitorTeam==Visitante)]
		PV=(Perdidos_Visitante['season'].groupby(Perdidos_Visitante.season).agg('count'))

		#Obtengo el n de goles marcado como Visitante
		Goles_Visitante=df[df.visitorTeam==Visitante]
		Goles_V=(Goles_Visitante['visitorGoals'].groupby(Goles_Visitante.season).agg('sum'))

		#Obtengo el total de goles de esos encuentros


		#contateno datos

		Estadisticas=pd.DataFrame(JL)
		Estadisticas.columns=['Jugados_L']
		Estadisticas['Ganados_L']=GL/Estadisticas['Jugados_L']
		Estadisticas['Empatados_L']=EL/Estadisticas['Jugados_L']
		Estadisticas['Perdidos_L']=PL/Estadisticas['Jugados_L']
		Estadisticas['Jugados_V']=JV
		Estadisticas['Ganados_V']=GV/Estadisticas['Jugados_V']
		Estadisticas['Empatados_V']=EV/Estadisticas['Jugados_V']
		Estadisticas['Perdidos_V']=PV/Estadisticas['Jugados_V']
		Estadisticas['Goles_L']=Goles_L/Estadisticas['Jugados_L']
		Estadisticas['Goles_V']=Goles_V/Estadisticas['Jugados_V']
		Estadisticas.fillna(0, inplace=True) #Reemplazo los NaN por 0
		Estadisticas['TotalGoles']=Goles_Encuentros

		#elimino aquellas filas que tienen nand
		Estadisticas.dropna(inplace=True)
		warning=''	
		if Estadisticas.shape[0]==0:
			print('Nunca se ha jugado este partido, por lo que no se puede realizar una predicción del resultado')
			warning='Nunca se ha jugado este partido, por lo que no se puede realizar una predicción del resultado'
			exit()
		elif Estadisticas.shape[0]<25:
			print('El número de registros es',Estadisticas.shape[0], ',por debajo de 25 encuentros las predicciones son menos seguras')
			warning='El número de partidos es ' + str(Estadisticas.shape[0])

		self.Model_Data=Estadisticas
		self.Datos_Corregidos=df
		self.Advertencia=warning

		return Estadisticas,df,warning

	def GeneroDic(self,JL,GL,Goles_L,JV,GV,Goles_V):

		Dic={}

		for valor in JL:
		
			season=valor[-1][0]
			equipo=valor[-1][1]
			jugados=valor[0]
			Dic[season + '-' + equipo]={'Jugados Local':jugados,
									'Ganados Local':0,
									'Goles Local':0,
									'Jugados Visitante':0,
									'Ganados Visitante':0,
									'Goles Visitante':0}


		for valor in GL:	
			season=valor[-1][0]
			equipo=valor[-1][1]
			Ganados=valor[0]
			Dic[season + '-' + equipo]['Ganados Local']=Ganados

		for valor in Goles_L:	
			season=valor[-1][0]
			equipo=valor[-1][1]
			Goles=valor[1]
			Dic[season + '-' + equipo]['Goles Local']=Goles

								
		for valor in JV:	
			season=valor[-1][0]
			equipo=valor[-1][1]
			Jugados=valor[0]
			Dic[season + '-' + equipo]['Jugados Visitante']=Jugados


		for valor in GV:	
			season=valor[-1][0]
			equipo=valor[-1][1]
			Ganados=valor[0]
			Dic[season + '-' + equipo]['Ganados Visitante']=Ganados


		for valor in Goles_V:	
			season=valor[-1][0]
			equipo=valor[-1][1]
			Goles=valor[1]
			Dic[season + '-' + equipo]['Goles Visitante']=Goles
				
		
		# print(Dic)
		return Dic
		
	def KNN(self):

		df=self.Datos_Corregidos

		Parametros=[
					[
						self.Equipo_Local['Total_Partidos_Ganados_Local']/self.Equipo_Local['Total_Partidos_Local'],
						self.Equipo_Local['Total_Goles_Local']/self.Equipo_Local['Total_Partidos_Local']
					],

					[
						self.Equipo_Visitante['Total_Partidos_Ganados_Visitante']/self.Equipo_Visitante['Total_Partidos_Visitante'],
						self.Equipo_Visitante['Total_Goles_Visitante']/self.Equipo_Visitante['Total_Partidos_Visitante']
					]
				]

		JL=(df.groupby(['season','localTeam']).agg('count'))
		JL['Indice'] = JL.index
		JL=JL.values.tolist()
		
		Ganados_local=df[(df.localGoals>df.visitorGoals)]
		GL=(Ganados_local.groupby(['season','localTeam']).agg('count'))
		GL['Indice'] = GL.index
		GL=GL.values.tolist()
		
		#Obtengo el n de goles marcado como local

		Goles_L=(df.groupby(['season','localTeam']).agg('sum'))

		Goles_L['Indice'] = Goles_L.index
		Goles_L=Goles_L.values.tolist()

		#obtengo partidos jugados como Visitante

		JV=(df.groupby(['season','visitorTeam']).agg('count'))
		JV['Indice'] = JV.index
		JV=JV.values.tolist()
		
		#Obtengo el ratio de partidos ganados fuera casa por temporada

		Ganados_Visitante=df[(df.localGoals<df.visitorGoals)]
		GV=(Ganados_Visitante.groupby(['season','visitorTeam']).agg('count'))
		GV['Indice'] = GV.index
		GV=GV.values.tolist()

		#Obtengo el n de goles marcado como Visitante
		Goles_V=(df.groupby(['season','visitorTeam']).agg('sum'))
		Goles_V['Indice'] = Goles_V.index
		Goles_V=Goles_V.values.tolist()

		#Genero diccionario de parametos por equipo
		Dic=self.GeneroDic(JL,GL,Goles_L,JV,GV,Goles_V)

		# df=df.values.tolist()
		# print(df)
		datos=[]
		for row in df.iterrows():
			#Actulizo la columna con la info correspondiente
		
			season=row[1][1]
			local=row[1][2]
			visitante=row[1][3]
			GolesLocal=row[1][4]
			GolesVisistante=row[1][5]
			PG_L=Dic[season + '-' + local]['Ganados Local'] / Dic[season + '-' + local]['Jugados Local']
			PG_V=Dic[season + '-' + visitante]['Ganados Visitante'] / Dic[season + '-' + visitante]['Jugados Visitante']
			G_L=Dic[season + '-' + local]['Goles Local'] / Dic[season + '-' + local]['Jugados Local']
			G_V=Dic[season + '-' + visitante]['Goles Visitante'] / Dic[season + '-' + visitante]['Jugados Visitante']
			
			#(Parametros) #[0][0] Ganados local [0][1] Goles local [1][0] Ganados Visitante [0][1] Goles Visitante
			Norma=LA.norm(Parametros)
			v2=[[PG_L,G_L],[PG_V,G_V]]
			distancia=LA.norm(ARR(Parametros)-ARR(v2))/Norma
			datos.append([distancia,GolesLocal+GolesVisistante])
		datos.sort()
		
		result={}
		total=0
		for dato in datos:

			if dato[0]<=0.05:
				
				k=round(dato[1],1)
				total+=1
				if k in result.keys():
					v=result[k]
					result[k]=v+1
				else: result[k]=1
			else:break
		
		print('Total partidos similares',total)
		warning='Total partidos similares ' + str(total)
		a=list(result.keys())
		a.sort()
		# print('Resultados encuentros similares')
		# for key in a:
			# try:
				# print(key,result[key]*100/total,'%')
			# except:
				# result={}
				# return result
			
		return result,warning


	def __init__(self):
		self.Equipos=self.get_equipos()
		self.Datos=self.get_data()