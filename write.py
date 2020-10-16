import csv
import Times
import os

def write_file(Destino,Nombre,headers,datos,tipo):
    t1=Times.Timer()
    
    
    if tipo=='w':
        with open(Destino+'/'+Nombre+'.csv',mode=tipo,newline='',encoding='utf8') as FicheroSalida:
            print('Escribiendo fichero',Nombre)
            FicheroSalida=csv.writer(FicheroSalida,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
            
            #Escribo cabecera
            FicheroSalida.writerow(headers)
            
            for linea in datos:
                FicheroSalida.writerow(linea)
        
        tf=round(Times.Timer(t1),3)
        print('Fichero',Nombre,'escrito con éxito en:',tf,'segundos')
        
    elif tipo=='a':
            
            ExisteFichero=os.path.isfile(Destino+'/'+Nombre+'.csv')
    
            with open(Destino+'/'+Nombre+'.csv',mode=tipo,newline='',encoding='utf8') as FicheroSalida:
            # with open(Destino+'/'+Nombre+'.csv',mode='r+',newline='',encoding='utf8') as FicheroSalida:
        
                FicheroSalida=csv.writer(FicheroSalida,delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
                
                #Escribo cabecera

                if ExisteFichero==False:
                    FicheroSalida.writerow(headers)
                    # print('el fichero es nuevo')
                    
                else: 
                    pass
                    # print('el fichero NO es nuevo')
                
                for linea in datos:
                    FicheroSalida.writerow(linea)