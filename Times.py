import time
#Si lo llamo sin argumentos me devuelve el tiempo
#Si le meto un argumentos me devuelve la resta del tiempo metido menos el tiempo actual
def Timer(*args):

	if len(args) > 0: return time.time()-args[0]		
	else: return time.time()