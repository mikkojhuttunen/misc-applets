"""Free-standing slab modes: TM0 / TE0 effective index and the share of the |E|² in the gas (Gamma), symmetric
cladding (air). TM0 here is the mode with H in the slab plane (E mostly normal to the membrane), the one the QUIET
membranes use. Functions: tm0, te0 (n_eff), gam (n_eff, Gamma). CLI: python design_calcs/slab.py [out.json]."""
import numpy as np
from scipy.optimize import brentq
def tm0(nc,ncl,t,lam):
    k0=2*np.pi/lam
    def f(neff):
        h=k0*np.sqrt(nc**2-neff**2);g=k0*np.sqrt(neff**2-ncl**2)
        return np.tan(h*t/2)-(nc**2/ncl**2)*g/h
    lo,hi=ncl+1e-6,nc-1e-6
    xs=np.linspace(lo,hi,4000);v=[f(x) for x in xs]
    for i in range(len(xs)-1):
        if v[i]*v[i+1]<0 and abs(v[i])<50 and abs(v[i+1])<50: return brentq(f,xs[i],xs[i+1])
def te0(nc,ncl,t,lam):
    k0=2*np.pi/lam
    def f(neff):
        h=k0*np.sqrt(nc**2-neff**2);g=k0*np.sqrt(neff**2-ncl**2)
        return np.tan(h*t/2)-g/h
    xs=np.linspace(ncl+1e-6,nc-1e-6,4000);v=[f(x) for x in xs]
    for i in range(len(xs)-1):
        if v[i]*v[i+1]<0 and abs(v[i])<50 and abs(v[i+1])<50: return brentq(f,xs[i],xs[i+1])
def gam(nc,t,lam,pol='TM'):
    ne=tm0(nc,1,t,lam) if pol=='TM' else te0(nc,1,t,lam)
    if ne is None: return None,None
    k0=2*np.pi/lam;h=k0*np.sqrt(nc**2-ne**2);g=k0*np.sqrt(ne**2-1)
    x=np.linspace(-t/2,t/2,4001)
    if pol=='TE':
        Ec=np.cos(h*x);Ea=np.cos(h*t/2)  # E_y continuous
        Ic=np.trapezoid(Ec**2,x);Io=Ea**2/(2*g)*2   # both sides: int exp(-2g d)=1/(2g) each
        return ne,Io/(Ic+Io)
    # TM: H_y ~ cos(hx), E_x ~ (1/eps) dH/dz-ish; use |E|^2 = |Ex|^2+|Ez|^2 with Ez ~ (neff/eps) H, Ex ~ (1/eps) dH/dx
    eps=nc**2
    Hc=np.cos(h*x);dH=-h*np.sin(h*x)*1.0
    Ec2=(neff_sq:=ne**2)*Hc**2/eps**2+dH**2/eps**2/(k0**2)  # arbitrary common factor
    Ha=np.cos(h*t/2)
    Eo2_int=(ne**2*Ha**2+g**2*Ha**2)/ (1.0**2)/(2*g)*2/(k0**2) if False else None
    # outside: H=Ha exp(-g d); Ez=ne*H (eps=1) ; Ex=g/k0 H -> |E|^2=(ne^2+g^2/k0^2)... in units k0=1:
    kk=k0
    Ec2=(ne**2*Hc**2+(dH/kk)**2)/eps**2
    Eo2=(ne**2+(g/kk)**2)*Ha**2
    Ic=np.trapezoid(Ec2,x);Io=Eo2/(2*g)*2
    return ne,Io/(Ic+Io)


def table():
    """The printed tables of the original script, as a dict."""
    out = {"Si_250nm_1p55": {"TM0": gam(3.48, 250e-9, 1.55e-6, 'TM'), "TE0": gam(3.48, 250e-9, 1.55e-6, 'TE')},
           "SiN_n2p0_1p55": {}, "SiN_n1p97_3p3": {}, "Si_3p42_3p3": {}}
    for t in (100, 150, 200, 250, 300, 400, 500, 600, 700):
        out["SiN_n2p0_1p55"][t] = {"TM0": gam(2.0, t * 1e-9, 1.55e-6, 'TM'), "TE0": gam(2.0, t * 1e-9, 1.55e-6, 'TE')}
    for t in (200, 300, 400, 500, 600, 800, 1000, 1200, 1500):
        out["SiN_n1p97_3p3"][t] = {"TM0": gam(1.97, t * 1e-9, 3.3e-6, 'TM')}
    for t in (250, 400, 600):
        out["Si_3p42_3p3"][t] = {"TM0": gam(3.42, t * 1e-9, 3.3e-6, 'TM')}
    return out


if __name__ == "__main__":
    import json, sys
    tab = table()
    a = tab["Si_250nm_1p55"]
    print('Si 250 nm TM0 1.55:', a["TM0"], ' TE0:', a["TE0"])
    print('SiN n=2.0 @1.55 um, TM0 / TE0 (n_eff, E-intensity fraction in gas)')
    for t, v in tab["SiN_n2p0_1p55"].items():
        print(t, 'nm TM0', tuple(round(x, 3) for x in v["TM0"]) if v["TM0"][0] else None, 'TE0', tuple(round(x, 3) for x in v["TE0"]) if v["TE0"][0] else None)
    print('SiN n=1.97 @3.3 um TM0')
    for t, v in tab["SiN_n1p97_3p3"].items():
        print(t, tuple(round(x, 3) for x in v["TM0"]) if v["TM0"][0] else None)
    if len(sys.argv) > 1:
        json.dump(tab, open(sys.argv[1], "w"), indent=1)
