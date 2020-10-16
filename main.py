import Apuestas_4 as Apuestas
import EstadisticasWeb
import numpy as np
# 34-
import streamlit as st

#Seleciono Equipos
Warning=[]

A=Apuestas.Apuestas()

st.header('Predicción de resultados')
Local=st.selectbox("Equipo local",A.Equipos['Equipo'].to_list())
Visitante=st.selectbox("Equipo Visitante",A.Equipos['Equipo'].to_list())

st.sidebar.button('Actualizar Datos')

Predecir=st.button('Predecir resultados')

if Predecir:

	A.set_encuentro(Local,Visitante)

	if A.Encuentro_posible==False:
		st.error('No puede jugar un equipo contra sí mismo!')
	else:
		#Obtengo estadisticas de los equipoas
		A.Estadisticas_Encuentro()
		A.DatosModelo()


		similares=A.KNN()

		pass
exit()
#Obtengo datos de los equipos
# urls=[]

# for equipo in [equipos[1],equipos[1]]:
# 	urls.append(Apuestas.getURL(equipo))
	
# ParametrosEntrada=[] # [[%partidos ganados local,media de goles local],[%partidos ganados Visitante,media de goles Visitante]]
# for i,url in enumerate(urls):
# 	ParametrosEntrada.append(EstadisticasWeb.estadisticas(equipos[i+1],url[0],i))
	
	
#Obtengo historico de los encuentros

# Estadisticas=Apuestas.DatosModelo(equipos[3],equipos[1],equipos[2])
# df=Estadisticas[1]
# Warning.append(Estadisticas[2])

similares=Apuestas.KNN(df,ParametrosEntrada)
Warning.append(similares[1])
#Genero Ruido

n=100
var=0.075
Ganados_L=np.random.normal(ParametrosEntrada[0][0], var, n).tolist()
Ganados_V=np.random.normal(ParametrosEntrada[1][0], var, n).tolist()
Goles_L=np.random.normal(ParametrosEntrada[0][1], var, n).tolist()
Goles_V=np.random.normal(ParametrosEntrada[1][1], var, n).tolist()


#Genero modelo y predigo resultado
resultados={}
for i in range(n):
	TotalGoles=Apuestas.RegresionLineal(Estadisticas[0],[Ganados_L[i],Ganados_V[i],Goles_L[i],Goles_V[i]])
	# print(TotalGoles,[Ganados_L[i],Ganados_V[i],Goles_L[i],Goles_V[i]])
	t=round(TotalGoles.item(),1)
	if resultados.get(t)==None:
		resultados[t]=1
	else:
		resultados[t]=resultados[t]+1
		

Intervalos=Apuestas.gen_intervalos(resultados) #Intervalos para predicciones ML
if similares[0]:
	Intervalos2=Apuestas.gen_intervalos(similares[0]) #Intervalos para partidos similares
else:Intervalos2=[]

# print(Intervalos2)

#Muestro grafico de probabilidades
Apuestas.Grafico_dist_goles(Estadisticas[0]['TotalGoles'],Estadisticas[0],equipos[1],equipos[2],resultados,Intervalos,Intervalos2,Warning)

#Get data from similares


