const {DitherLab:D}=require('./eng2.js');
const fs=require('fs');
const R=+process.argv[2], dB=+process.argv[3], cellName=process.argv[4], nSeed=+process.argv[5], out=process.argv[6];
const alpha=dB/4.3429448190325175/0.01;
const BUDn={ n_samp: 8192, n_u: 24, n_v: 72, n_side: 12, poin_s: 16, poin_p: 10, poin_hits: 700, map_n: 7 };
const LAM=1653.7262586923987e-9;
const common=Object.assign({wavelength:LAM,species:'CH4',n_eff:2,R_mirror:R,alpha_m:alpha,gamma:0.2,P_in:1e-3,dither:2e-3,waveform:'sine',f_theta:1000,
  dlam:10**1.3*1e-12,lwave:'triangle',f_lambda:1297,stray:1,f_parasitic:1e-6,speckle_avg:1,ctr_dn:0},BUDn);
const inc={n_inc:16,inc_radius:50e-6,inc_dn:0.01,inc_sign:'both'};
const circle=Object.assign({},common,{cell:'circle',c_radius:2e-3,n_facets:24,facet_curv:0,c_theta:7.5*Math.PI/180,c_port:150e-6,c_waist:40e-6,out_frac:0.458,same_port:0,f_tilt_rms:0,f_curv_rms:0,f_seed:1,f_sel:5,f_sel_tilt:0,f_sel_curv:0,inc_ring:1},inc);
const herriott=Object.assign({},common,{cell:'herriott',L:10e-3,W:1.5e-3,N:14,M:3,distinct:1,port_width:0,walls:1,retune:1,dR1:0,dR2:0,tilt1:0,tilt2:0,eps4:0,ripple_a:0,ripple_period:0.5e-3},inc);
const base=cellName==='circle'?circle:herriott;
const eps=1e-6;
const keep=r=>({grains:r.grains,T:r.T,L_mean:r.L_mean,V_theta:r.V_theta,V_lambda:r.V_lambda,V_joint:r.V_joint,V_floor:r.V_floor,V_floor_both:r.V_floor_both,n_pass:r.n_pass});
const res={R,dB,cell:cellName,regular:null,chaotic:[]};
const t0=Date.now();
try{res.regular=keep(D.analyse(D.regular(Object.assign({},base,{inc_seed:1})),{full:false}));}catch(e){res.regular={error:String(e.message)};}
console.log('regular',JSON.stringify(res.regular),((Date.now()-t0)/1000).toFixed(1)+'s');
for(let s=1;s<=nSeed;s++){
  try{res.chaotic.push(Object.assign({seed:s},keep(D.analyse(Object.assign({},base,{inc_seed:s}),{full:false}))));}catch(e){res.chaotic.push({seed:s,error:String(e.message)});}
  if(s%5===0){console.log('seed',s,((Date.now()-t0)/1000).toFixed(0)+'s');fs.writeFileSync(out,JSON.stringify(res));}
}
fs.writeFileSync(out,JSON.stringify(res));console.log('done');
