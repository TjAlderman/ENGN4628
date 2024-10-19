from scipy.optimize import curve_fit
import numpy as np

def polynomial_11th_order(x,a,b,c,d,e,f,g,h,i,j,k,l,m):
    return a*x+b*x**2+c*x**3+d*x**4+e*x**4+f*x**5+g*x**6+h*x**7+i*x**8+j*x**9+k*x**10+l*x**11+m

def exponential(x, a, b, c, d):
    return c*(np.exp(a*(x-b))-a*(x-b)-1)+d
    
def fit(x,y,mode='all'):
    modes = {"polynomial":[polynomial_11th_order],
              "exponential":[exponential],
              "all":[polynomial_11th_order,exponential]}
    params = []
    rmses = []
    fns = []
    for fn in modes[mode]:
        param, cov = curve_fit(fn,x,y,maxfev=1000000)
        rmse = np.sqrt(np.mean((y - fn(x, *param))**2))
        params.append(param)
        rmses.append(rmse)
        fns.append(fn)
    rmses = np.array(rmses)
    idx = np.argmin(rmses)
    return params[idx].tolist(), fns[idx]

def fitted(x,params,fn):
    return fn(x,*params)



