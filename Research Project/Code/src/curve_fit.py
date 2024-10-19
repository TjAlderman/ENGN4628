from scipy.optimize import curve_fit
import numpy as np

def polynomial_11th_order(x,a,b,c,d,e,f,g,h,i,j,k,l,m):
    return a*x+b*x**2+c*x**3+d*x**4+e*x**4+f*x**5+g*x**6+h*x**7+i*x**8+j*x**9+k*x**10+l*x**11+m

def exponential(x, a, b, c, d):
    return c*(np.exp(a*(x-b))-a*(x-b)-1)+d
    
def fit(x,y):
    params1, cov1 = curve_fit(polynomial_11th_order,x,y,maxfev=1000000)
    rmse1 = np.sqrt(np.mean((y - polynomial_11th_order(x, *params1))**2))
    params2, cov2 = curve_fit(exponential,x,y,maxfev=1000000)
    rmse2 = np.sqrt(np.mean((y - exponential(x, *params2))**2))
    # return params2.tolist(), "exponential"
    return params1.tolist(), "polynomial" # Temporary hack
    if rmse1>rmse2:
        return params1.tolist(), "polynomial"
    else:
        return params2.tolist(), "exponential"

def fitted(x,params,mode="polynomial"):
    if mode=="polynomial":
        return polynomial_11th_order(x,*params)
    elif mode=="exponential":
        return exponential(x,*params)
    else:
        raise Exception(f"Error! Unrecognised mode: {mode}")



