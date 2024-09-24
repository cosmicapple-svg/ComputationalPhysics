"""
Módulo para el capítulo 5 Giordano: Potenciales y campos

Las siguientes funciones son utilizadas para resolver numéricamente ecuaciones diferenciales parciales
como:
    - Laplace
    - Poisson
"""

import numpy as np
import matplotlib.pyplot as plt

def Discretiza(X, Y, N=[10, 10]):
    """Discretiza el plano en la región x[x0, x1], Y en [y0, y1] en secciones
        N = [nx, ny]"""
    
    x = np.linspace(X[0], X[1], N[0])
    y = np.linspace(Y[0], Y[1], N[1])
    X, Y = np.meshgrid(x, y)
    V = np.zeros([N[0], N[1]], dtype=float)
    
    return  X, Y, V

def CC1(V, V0, V1, Vx):
    """
    Condiciones de contorno para la caja metálica.
    """
    #Establecemos las condiciones de contorno
    indicesCC = []
    nx = np.shape(V)[1]
    
    for i in range(nx):
        V[i, 0] =  V0          #V(x, -1) = -1
        indicesCC.append((i, 0))
        V[i, nx-1] = V1         #V(x, 1 ) =  1
        indicesCC.append((i, nx-1))
    
        V[0, i] = Vx(i)        #V(x, 1) =  x
        indicesCC.append((0, i))
        V[nx-1, i] = Vx(i)     #V(x, -1) = x
        indicesCC.append((nx-1, i))
    
    return V, indicesCC

def RelaxationJacobi(V, indicesCC, err=1e-7, imax=1000):
    
    diff = 0.; iteracion=0   # Inicializamos
    V_next = V_old = V.copy()
    nx = np.shape(V)[1]; ny = np.shape(V)[0]

    #Este ciclo while realiza el método de Jacobi
    while True:
        
        # Los límites de los for se les quitan las 'orillas' para que cada punto tenga vecinos.
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                #Filtramos los puntos no definidos por las CC
                if (i, j) in indicesCC:
                    pass
                else: 
                    tmp= V_old[i, j]
                    #Se promedia como el método lo especifica
                    V_next[i, j] = (V_old[i+1, j] + V_old[i-1, j] + V_old[i, j-1] + V_old[i, j+1])/4.
                    #print(V_next[i, j], V_old[i, j])
                    #Cada punto contribuye con error, el cual se va sumando 
                    diff += np.abs( tmp - V_next[i, j])
        diff /= (2*nx +1)**2
        
        iteracion += 1
        if diff< err: #¿La diferencia de potencial es menor al mínimo establecido?
            print("Tolerancia de error mínima lograda. Err:", diff)
            print("Iteraciones: ", iteracion)
            break
            
        elif iteracion >= imax: #¿Se alcanzó el límite de iteraciones?
            print("Límite de iteraciones alcanzado:", iteracion)
            break
            
        else:
            #Redefinimos los arreglos para que funcione la iteración (el nuevo es el viejo en la sig. iteración)
            V_old = V_next.copy()
            diff = 0
        
    return V_next, iteracion, diff

def Surface3D(X, Y, V_next, save=False, savename='fig.png'):
    #Ploteamos el potencial 
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes(projection='3d')
    plt.style.use('dark_background')
    ax.plot_surface(X, Y, V_next, cmap ='inferno')
    ax.grid(True, color='dimgray') 
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('V(x, y)')
    if save==True:
        plt.savefig(savename, dpi=300)
    else: pass
    plt.show()
    
def Potencial(V, save=False, savename='blabla.png'):
    fig = plt.figure(figsize=(5, 5))
    ax = plt.axes()
    plt.style.use('dark_background')
    ax.imshow(V, cmap='bone')
    if save == True:
        plt.savefig(savename, dpi=300)
    else:
        pass
    plt.show()

def EField(V):
    #Electric field (x direction)
    nx = len(V[0, :]); ny = len(V[:, 0])
    
    Ex = np.zeros([nx, nx]); Ey = np.zeros([ny, ny])
    #Se calcula derivada con diferencia centrada
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            Ex[i, j] = -(V[i+1, j] - V[i-1, j])/(2*(2/nx)) 
            Ey[i, j] = -(V[i, j+1] - V[i, j-1])/(2*(2/ny)) 
    
    return Ex, Ey

def PlotEField(x, y, Ex, Ey, save = False, savefig='Field.png')  :           
    #Graficación
    color = np.hypot(Ex, Ey) #Escala de colores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.style.use('dark_background')
    ax.streamplot(x, y, Ey, Ex, color = color, cmap=plt.cm.plasma, density=1)
    ax.grid(True, color='dimgray')
    
    if save == True:
        plt.savefig(savefig, dpi=300)
    else:
        pass
    plt.show()

    
def RelaxationGaussSeidel(V, indicesCC, err=1e-5, imax=10000):
    
    diff = 0.; iteracion=0   # Inicializamos
    V_next = V_old = V.copy()
    nx = np.shape(V)[1]; ny = np.shape(V)[0]

    #Este ciclo while realiza el método de Jacobi
    while True:
        
        # Los límites de los for se les quitan las 'orillas' para que cada punto tenga vecinos.
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                #Filtramos los puntos no definidos por las CC
                if (i, j) in indicesCC:
                    pass
                else: 
                    #Se promedia como el método lo especifica
                    tmp = V_old[i, j]
                    V_next[i, j] = (V_next[i+1, j] + V_next[i-1, j] + V_next[i, j+1] + V_next[i, j-1])/4.
                    #print(V_next[i, j], V_old[i, j])
                    #Cada punto contribuye con error, el cual se va sumando 
                    diff += np.abs( tmp - V_next[i, j])
        diff /= (2* nx +1)**2
        
        iteracion += 1
        if diff< err: #¿La diferencia de potencial es menor al mínimo establecido?
            print("Tolerancia de error mínima lograda. Err:", diff)
            print("Iteraciones: ", iteracion)
            break
            
        elif iteracion >= imax: #¿Se alcanzó el límite de iteraciones?
            print("Límite de iteraciones alcanzado:", iteracion)
            break
            
        else:
            #Redefinimos los arreglos para que funcione la iteración (el nuevo es el viejo en la sig. iteración)
            V_old = V_next.copy()
        
    return V_next, iteracion, diff