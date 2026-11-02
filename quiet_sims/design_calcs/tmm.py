"""Effective-index trench DBR transfer matrix (s-pol), as received; the main search runs only as a script
(python design_calcs/tmm.py). qs/mirror.py is the vectorised, extended version used by the QUIET steps."""
import numpy as np
def refl(ns,N,dg,dt,sinth,lam=1.0):
    # s-pol (E out of plane = slab TM0), slab | (gap, tooth)xN | gap | air ; layers: n_list, d_list
    ns_=[1.0 if i%2==0 else ns for i in range(2*N)]  # gap, tooth,...
    nl=[]+ns_+[1.0]; dl=[]
    for i in range(N): dl+= [dg,dt]
    dl+=[dg]
    k0=2*np.pi/lam
    out=[]
    for s in np.atleast_1d(sinth):
        sin=ns*s
        def cz(n):
            v=n*n-sin*sin+0j
            return np.sqrt(v)
        M=np.eye(2,dtype=complex)
        for n,d in zip(nl,dl):
            c=cz(n);ph=k0*c*d
            Mi=np.array([[np.cos(ph),-1j*np.sin(ph)/c],[-1j*c*np.sin(ph),np.cos(ph)]])
            M=M@Mi
        c0=cz(ns);cs=cz(1.0)
        B=M@np.array([1,cs]);Cc=B[1]
        r=(c0*B[0]-Cc)/(c0*B[0]+Cc);out.append(abs(r)**2)
    return np.array(out)
def best(ns,N):
    thc=np.arcsin(1/ns);ang=np.linspace(0,0.97*np.sin(thc)/ns*ns,60)  # sin(theta in slab) from 0 to 0.97/ns
    sins=np.linspace(0,0.97/ns,70)
    bestv=(0,None)
    for mg in (1,3,5):
        for mt in (1,3,5):
            for thd in np.deg2rad(np.arange(0,46,3)):
                s0=np.sin(thd)
                if s0>0.97/ns: continue
                # quarter-wave at design angle in each layer: n cos(theta_l) d = m lam/4 ; k_z of layer in units
                cg=np.sqrt(1-(ns*s0)**2+0j).real; ct=np.sqrt(ns**2-(ns*s0)**2).real
                dg=mg/4/cg; dt=mt/4/ct
                R=refl(ns,N,dg,dt,sins)
                v=R.min()
                if v>bestv[0]: bestv=(v,(mg,mt,np.rad2deg(thd),R[0],R[len(R)//2],R[-1]))
    return bestv,np.rad2deg(np.arcsin(1/ns))
if __name__ == "__main__":
    for ns,name in ((1.36,'SiN 400 nm TM0 (n_eff 1.36)'),(1.25,'SiN thinner (1.25)'),(1.5,'SiN thick (1.5)'),(2.0,'Si 250 nm TM0 (n_eff 2.0)')):
        print(name)
        for N in (3,4,5,6,8,10):
            (v,info),thc=best(ns,N)
            print('  N=%2d  worst-case R over 0..%.0f%% of crit angle (%.0f deg): %.4f  | mg,mt,design deg,R(0),R(mid),R(edge)=%s'%(N,97,thc,v,tuple(round(float(x),4) for x in info)))
