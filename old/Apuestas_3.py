#apuestas

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from sklearn import preprocessing
#variables 

# AVG goles metidos local año por partido, 
# AVG goles encajados local año por partido,
# AVG goles metidos Visitante año por partido, 
# AVG goles encajados Visitante año por partido,
# ranking tabla local
# ranking tabla visitante
# Resultado (0: 1-0,1:1-1,2:2-1,3:otro)

Ruta_equipos='Datos/equipos.csv'
Ruta_Datos='Datos/FMEL_Dataset.csv'


def Grafico_dist_goles(TGoles,Estadisticas,Local,Visitante,TotalGoles_prediccion,leyenda):
	hist_goles=pd.DataFrame()
	hist_goles['Pgoles']=(TGoles.groupby(Estadisticas.TotalGoles).agg('count'))/TGoles.groupby(Estadisticas.TotalGoles).agg('count').sum()

	# fig = plt.figure()
	# ax1 = plt.axes()
	# ax2 = plt.axes()
	acum=0
	x=[]
	y1=[]
	y2=[]
	for index, row in hist_goles.iterrows():
		acum+=row['Pgoles']*100
		x.append(index)
		y1.append(row['Pgoles']*100)
		y2.append(acum)
		plt.text(index-0.1,acum+2,str(round(acum,1))+'%')
		plt.text(index,row['Pgoles']*100-2,str(round(row['Pgoles']*100,1))+'%')
	
	x1=[]
	y3=[]
	i=0
	for valor in sorted(TotalGoles_prediccion.keys()):
		# print(valor)
		x1.append(valor)
		y3.append(TotalGoles_prediccion[valor]*100 / sum(TotalGoles_prediccion.values()))
		if i==0:
			plt.text(valor-0.3,TotalGoles_prediccion[valor]*100 / sum(TotalGoles_prediccion.values()),'mín:'+str(valor))
		if i==len(TotalGoles_prediccion)-1:
			plt.text(valor+0.1,TotalGoles_prediccion[valor]*100 / sum(TotalGoles_prediccion.values()),'máx:'+str(valor))
		i+=1
	
	plt.text(0,-15,'Mínimo:'+str(leyenda[0])+' Media:'+str(leyenda[1])+' Máximo:'+str(leyenda[2]))
	plt.plot(x,y1,color='skyblue',label='Prob')
	plt.plot(x,y2, color='lightpink',label='Prob Acumulada')
	plt.plot(x1,y3, color='gold',label='Distribución de goles')

	plt.legend(loc='best',shadow=True,fontsize='small')

	plt.fill_between(x, y2,color="lightpink", alpha=0.4)
	plt.fill_between(x, y1,color="skyblue", alpha=0.6)
	plt.fill_between(x1, y3,color="gold", alpha=0.5)
	plt.grid()
	plt.title('Distribución de goles histórica entre: ' +Local +' vs ' + Visitante)
	plt.ylabel('Probabilidad')
	plt.xlabel('# de goles totales')
	# plt.axvline(x=TotalGoles_prediccion,color='r', alpha=0.6)
	# plt.text(TotalGoles_prediccion+0.1,90,"Previsión: "+str(TotalGoles_prediccion.round(2))+" goles marcados",fontsize=8)
	# plt.text(TotalGoles_prediccion+0.1,90,"Previsión: "+str(TotalGoles_prediccion.values)+" goles marcados",fontsize=8)
	plt.show()


def RegresionLineal(df,datos):

	#Selecciona los datos que quieras con las cabeceras
	cdf = df[['Ganados_L','Empatados_L','Perdidos_L','Ganados_V','Empatados_V','Perdidos_V','Goles_L','Goles_V','TotalGoles']]

	#Divido los datos en train y test

	# msk = np.random.rand(len(df)) < 0.8
	# train = cdf[msk]
	# test = cdf[~msk]

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
	# print('El total de goles previstos es de:',TotalGoles)

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

	id_Equipos=[int(ele) for ele in encuentro.split('-')]

	Local=df_Equipos['Equipo'][id_Equipos[0]]
	Visitante=df_Equipos['Equipo'][id_Equipos[1]]
	
	return [id_Equipos,Local,Visitante,df]
	
def getURL(Nombre):

	Equipos_Lista = pd.read_csv(Ruta_equipos)
	df=pd.DataFrame(Equipos_Lista)
	Datos_Encuentros=df[df.Equipo==Nombre]
	return Datos_Encuentros.urlData.values

def DatosModelo(df,Local,Visitante):

	
	# Quito columnas inutiles
    df.drop('round', axis='columns',inplace=True)
    df.drop('date', axis='columns',inplace=True)
    df.drop('timestamp', axis='columns',inplace=True)
    df.drop('division', axis='columns',inplace=True)
    
    #Creo una columna Año
    df['año']=df['season'][:4]
    
    print(df)
    exit()
    #Obtengo datos de estadisticas historicas

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
        
    if Estadisticas.shape[0]==0:
        print('Nunca se ha jugado este partido, por lo que no se puede realizar una predicción del resultado')
        exit()
    elif Estadisticas.shape[0]<25:
        print('El número de registros es',Estadisticas.shape[0], ',por debajo de 25 encuentros las predicciones son menos seguras')
        

    #Entreno el modelo de regresión

    return Estadisticas
    exit()
    

    RegresionLineal(Estadisticas,DatosPrediction)



    #Predigo el resultado
    Grafico_dist_goles(Estadisticas['TotalGoles'])

    #encuentro los goles del 50% de probabilidad acumulada
