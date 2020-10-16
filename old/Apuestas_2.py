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

def get_stat(id,atr):
	
	#Abro equipo estadisticas equipos
	
	Equipos_stat = pd.read_csv(Ruta_equipos)
	df=pd.DataFrame(Equipos_stat)
	if atr=='L':
		DatosL=df[df.id==id]
		return DatosL['ratio ganados local'],DatosL['media goles local']
	else:
		local=0
		Vis=1

def Grafico_dist_goles(TGoles):
	hist_goles=pd.DataFrame()
	hist_goles['Pgoles']=(TGoles.groupby(Estadisticas.TotalGoles).agg('count'))/TGoles.groupby(Estadisticas.TotalGoles).agg('count').sum()

	fig = plt.figure()
	ax1 = plt.axes()
	ax2 = plt.axes()
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

	plt.plot(x,y1,color='skyblue',label='Prob')
	plt.plot(x,y2, color='lightpink',label='Prob Acumulada')
	plt.legend(loc='best',shadow=True,fontsize='small')

	plt.fill_between(x, y2,color="lightpink", alpha=0.4)
	plt.fill_between(x, y1,color="skyblue", alpha=0.6)
	plt.grid()
	plt.title('Distribución de goles histórica entre: ' +Local +' vs ' + Visitante)
	plt.ylabel('Probabilidad')
	plt.xlabel('# de goles totales')
	plt.show()


def RegresionLineal(df):

	#Selecciona los datos que quieras con las cabeceras
	cdf = df[['Ganados_L','Empatados_L','Perdidos_L','Ganados_V','Empatados_V','Perdidos_V','Goles_L','Goles_V','TotalGoles']]

	#Divido los datos en train y test

	msk = np.random.rand(len(df)) < 0.8
	train = cdf[msk]
	test = cdf[~msk]

	#Creo el modelo lineal 

	from sklearn import linear_model
	regr = linear_model.LinearRegression()
	# x = np.asanyarray(train[['Ganados_L','Empatados_L','Perdidos_L','Ganados_V','Empatados_V','Perdidos_V','Goles_L','Goles_V']])
	x = np.asanyarray(train[['Ganados_L','Ganados_V','Goles_L','Goles_V']])
	y = np.asanyarray(train[['TotalGoles']])
	regr.fit (x, y)
	# The coefficients
	print ('Coefficients: ', regr.coef_)

	#compruebo resultados

	y_hat= regr.predict(test[['Ganados_L','Ganados_V','Goles_L','Goles_V']])
	# y_hat= regr.predict(test[['Ganados_L','Empatados_L','Perdidos_L','Ganados_V','Empatados_V','Perdidos_V','Goles_L','Goles_V']])
	x = np.asanyarray(test[['Ganados_L','Ganados_V','Goles_L','Goles_V']])
	# x = np.asanyarray(test[['Ganados_L','Empatados_L','Perdidos_L','Ganados_V','Empatados_V','Perdidos_V','Goles_L','Goles_V']])
	y = np.asanyarray(test[['TotalGoles']])
	# print("Residual sum of squares: %.2f"
		  # % np.mean((y_hat - y) ** 2))
	aciertos=0
	print('real','pred')
	for i in range(len(y)):
		if y_hat[i]>1.5 and y[i]>1:
			a='Acierto'
			aciertos+=1
		else:a='Fallo'
		print(y[i],y_hat[i].round(),a)

	print('Total veces acertadas que se marca más de 1.5 goles:',aciertos,', falladas:',
		len(y)-aciertos,'ratio:',round(aciertos*100/(len(y)),1),'%')
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % regr.score(x, y))


#Leo datos de los equipos

Ruta_equipos='Datos/equipos.csv'
Ruta_Datos='Datos/FMEL_Dataset.csv'

Equipos_Lista = pd.read_csv(Ruta_equipos)
Datos=pd.read_csv(Ruta_Datos)

#Obtengo los valores unicos


df=pd.DataFrame(Datos)
Equipos=Datos.localTeam.unique()
df_Equipos=pd.DataFrame(Equipos)
# print(df_Equipos)
pd.set_option('display.max_rows', df_Equipos.shape[0]+1)
print(df_Equipos.sort_values(by=[0]))

encuentro= input('Escribe el encuentro en formato id - id: ')

Equipos=[int(ele) for ele in encuentro.split('-')]

Local=df_Equipos[0][Equipos[0]]
Visitante=df_Equipos[0][Equipos[1]]

print('Juegan el', Local,'contra el', Visitante)

#Quito columnas inutiles

df.drop('round', axis='columns',inplace=True)
df.drop('date', axis='columns',inplace=True)
df.drop('timestamp', axis='columns',inplace=True)
df.drop('division', axis='columns',inplace=True)

#Obtengo datos de estadisticas historicas

Datos_Encuentros=df[(df.localTeam==Local) & (df.visitorTeam==Visitante)]
Goles_Encuentros=(Datos_Encuentros['visitorGoals'].groupby(Datos_Encuentros.season).agg('sum'))+(Datos_Encuentros['localGoals'].groupby(Datos_Encuentros.season).agg('sum'))
# print(Goles_Encuentros)

#Obtengo goles por partido del equipo local por año

# EstadisticasLocal=df[df.localTeam==Local]


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
	
elif Estadisticas.shape[0]<25:
	print('El número de registros es',Estadisticas.shape[0], ',por debajo de 25 encuentros las predicciones son menos seguras')
	

#Entreno el modelo de regresión

#pido datos de los equipos:

# ['Ganados_L','Ganados_V','Goles_L','Goles_V']

#Leo los datos del los equipos locales y visitantes

get_stat(Equipos[0],'L')
exit()
get_stat(Equipos[1],'V')

RegresionLineal(Estadisticas,DatosPrediction)



#Predigo el resultado
Grafico_dist_goles(Estadisticas['TotalGoles'])

#encuentro los goles del 50% de probabilidad acumulada
