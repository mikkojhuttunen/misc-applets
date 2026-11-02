const {DitherLab:D,GasEngine:G}=require('./eng_loss.js');
const fs=require('fs');
const data=JSON.parse(fs.readFileSync('fig3data.json'));
const base=Object.assign({},data.chaos.circle,{cell:'circle',n_inc:16,c_theta:7.5*Math.PI/180,R_mirror:0.99,alpha_m:0.1/4.3429448/0.01,out_frac:0.458});
const neffs=[1.2,1.36,1.5,1.7,2.0,2.7];
const acc={};neffs.forEach(n=>acc[n]={hits:0,sub:0,rays:0,raysAny:0,wsum:0,wsub:0});
let nh=0,sinHist=new Array(50).fill(0);
for(const seed of [1,2,3,4,5,6,7,8]){
  for(const reg of [false]){
  const o=Object.assign({},base,{inc_seed:seed});
  const S=D.setup(o),c=S.mc.cell,pw=c.meta.portW;
  const nB=400;
  const rays=G.beamRays(c,c.sIn,S.theta,S.q0,S.lamM,12);
  for(const st of rays){
    const r=G.marchRay(S.mc,st,nB,c.sIn,c.sOut,pw,'both');
    if(r.exit!==1)continue;
    const n=r.nWalls;const w=Math.pow(0.99,n-1)*Math.exp(-(0.1/4.3429448/0.01)*r.WG[n-1]);
    for(const ne of neffs){const a=acc[ne];a.rays++;let any=false;
      for(let i=0;i<n;i++){const s=Math.abs(r.WC[i]);a.hits++;a.wsum+=w;if(s<1/ne){a.sub++;a.wsub+=w;any=true;}}
      if(any)a.raysAny++;}
    for(let i=0;i<n;i++){const s=Math.abs(r.WC[i]);sinHist[Math.min(49,Math.floor(s*50))]++;nh++;}
  }}
}
console.log('n_eff  frac of bounces below critical angle | frac of detected rays with >=1 sub-critical bounce');
for(const ne of neffs){const a=acc[ne];console.log(ne,'crit angle',(Math.asin(1/ne)*180/Math.PI).toFixed(0)+'deg',(a.sub/a.hits*100).toFixed(1)+' %',(a.raysAny/a.rays*100).toFixed(1)+' %');}
require('fs').writeFileSync('sinhist.json',JSON.stringify(sinHist.map(x=>x/nh)));
