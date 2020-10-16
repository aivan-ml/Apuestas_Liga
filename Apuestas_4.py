import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from sklearn import preprocessing
import time

Ruta_equipos='Datos/equipos.csv'
Ruta_Datos='Datos/FMEL_Dataset.csv'

def gen_intervalos(Diccionario):

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

def CorrijoDatos(df):

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

def RegresionLineal(df,datos):

	#Creo el modelo lineal 

	from sklearn import linear_model
	regr = linear_model.LinearRegression()
	# x = np.asanyarray(train[['Ganados_L','Empatados_L','Perdidos_L','Ganados_V','Empatados_V','Perdidos_V','Goles_L','Goles_V']])
	x = np.asanyarray(df[['Ganados_L','Ganados_V','Goles_L','Goles_V']])
	y = np.asanyarray(df[['TotalGoles']])
	regr.fit (x, y)
	# The coefficients
	# print ('Coefficients: ', regr.coef_)

	#compruebo resultados
	TotalGoles=regr.predict([datos])
	# y_hat= regr.predict(test[['Ganados_L','Ganados_V','Goles_L','Goles_V']])
	return TotalGoles

def get_players():

	#Leo datos de los equipos

	Datos=pd.read_csv(Ruta_Datos)
	Datos_Equipos=pd.read_csv(Ruta_equipos)
	
	#Obtengo los valores unicos


	df=pd.DataFrame(Datos)
	df_Equipos=pd.DataFrame(Datos_Equipos)
	df_Equipos=df_Equipos[['Equipo']].sort_values(by=['Equipo'])
	# Equipos=Datos.localTeam.unique()
	Equipos=Datos_Equipos.Equipo
	# df_Equipos=pd.DataFrame(Equipos)
	
	pd.set_option('display.max_rows', df_Equipos.shape[0]+1)
	# print(df_Equipos[['Equipo']].sort_values(by=['Equipo']))
	print(df_Equipos['Equipo'])
	# print(df_Equipos.sort_values(by=[0]))
	# print(df_Equipos)
	encuentro= input('Escribe el encuentro en formato id - id: ')
	try:
		id_Equipos=[int(ele) for ele in encuentro.split('-')]
	
	
		Local=df_Equipos['Equipo'][id_Equipos[0]]
		Visitante=df_Equipos['Equipo'][id_Equipos[1]]
		
		return [id_Equipos,Local,Visitante,df]
	except:
		print('Error en los ids')
		exit()
	
def getURL(Nombre):

	Equipos_Lista = pd.read_csv(Ruta_equipos)
	df=pd.DataFrame(Equipos_Lista)
	Datos_Encuentros=df[df.Equipo==Nombre]
	return Datos_Encuentros.urlData.values

def DatosModelo(df,Local,Visitante):

	
	df=CorrijoDatos(df)

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
		exit()
	elif Estadisticas.shape[0]<25:
		print('El número de registros es',Estadisticas.shape[0], ',por debajo de 25 encuentros las predicciones son menos seguras')
		warning='El número de partidos es ' + str(Estadisticas.shape[0])

	return Estadisticas,df,warning

def GeneroDic(JL,GL,Goles_L,JV,GV,Goles_V):

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
	
def KNN(df,Parametros):

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
	Dic=GeneroDic(JL,GL,Goles_L,JV,GV,Goles_V)

	df['PG_L']=np.nan
	df['PG_V']=np.nan
	df['G_L']=np.nan
	df['G_V']=np.nan
	df=df.values.tolist()
	datos=[]
	for row in df:
		#Actulizo la columna con la info correspondiente
	
		season=row[1]
		local=row[2]
		visitante=row[3]
		GolesLocal=row[4]
		GolesVisistante=row[5]
		PG_L=Dic[row[1] + '-' + row[2]]['Ganados Local'] / Dic[row[1] + '-' + row[2]]['Jugados Local']
		PG_V=Dic[row[1] + '-' + row[3]]['Ganados Visitante'] / Dic[row[1] + '-' + row[3]]['Jugados Visitante']
		G_L=Dic[row[1] + '-' + row[2]]['Goles Local'] / Dic[row[1] + '-' + row[2]]['Jugados Local']
		G_V=Dic[row[1] + '-' + row[3]]['Goles Visitante'] / Dic[row[1] + '-' + row[3]]['Jugados Visitante']
		
		#(Parametros) #[0][0] Ganados local [0][1] Goles local [1][0] Ganados Visitante [0][1] Goles Visitante
		Norma=(Parametros[0][0]**2+Parametros[0][1]**2+Parametros[1][0]**2+Parametros[1][1]**2)**0.5
		distancia=(((PG_L-Parametros[0][0])**2+
					(G_L-Parametros[0][1])**2+
					(PG_V-Parametros[1][0])**2+
					(G_V-Parametros[1][1])**2)**0.5)/Norma
		
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



