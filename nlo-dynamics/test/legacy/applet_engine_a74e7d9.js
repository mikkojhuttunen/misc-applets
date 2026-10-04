/* Frozen verbatim copy of the numerical engine of
   physics-applets/laserquiz-rewards/mode-locking-explorer.html at commit a74e7d9
   (lines 393-497: mlRandn, fft, mlStep, mlAnalyze, plus ML_Nt/ML_omega setup).
   Used ONLY as a regression oracle for nlo-engine.js. Do not edit. */
'use strict';
const ML_Nt = 256;
const ML_omega = new Float64Array(ML_Nt);
for(let k=0;k<ML_Nt;k++){ const kk = k<=ML_Nt/2? k : k-ML_Nt; ML_omega[k] = 2*Math.PI*kk/ML_Nt; }
function mlRandn(){
  let u=0,v=0; while(u===0)u=Math.random(); while(v===0)v=Math.random();
  return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
}

// iterative radix-2 FFT, in place
function fft(re, im, inverse){
  const N = re.length;
  for(let i=1,j=0;i<N;i++){
    let bit=N>>1;
    for(;j&bit;bit>>=1) j^=bit;
    j^=bit;
    if(i<j){ let t=re[i];re[i]=re[j];re[j]=t; t=im[i];im[i]=im[j];im[j]=t; }
  }
  for(let len=2;len<=N;len<<=1){
    const ang = (inverse?2:-2)*Math.PI/len;
    const wr0 = Math.cos(ang), wi0 = Math.sin(ang);
    for(let i=0;i<N;i+=len){
      let curWr=1, curWi=0;
      for(let k=0;k<len/2;k++){
        const ur=re[i+k], ui=im[i+k];
        const vr=re[i+k+len/2]*curWr - im[i+k+len/2]*curWi;
        const vi=re[i+k+len/2]*curWi + im[i+k+len/2]*curWr;
        re[i+k]=ur+vr; im[i+k]=ui+vi;
        re[i+k+len/2]=ur-vr; im[i+k+len/2]=ui-vi;
        const nwr = curWr*wr0 - curWi*wi0, nwi = curWr*wi0 + curWi*wr0;
        curWr=nwr; curWi=nwi;
      }
    }
  }
  if(inverse){ for(let i=0;i<N;i++){ re[i]/=N; im[i]/=N; } }
}

function mlStep(re, im, q, p){
  const Nt = ML_Nt;
  let E=0; for(let i=0;i<Nt;i++) E += re[i]*re[i]+im[i]*im[i];

  // SPM
  for(let i=0;i<Nt;i++){
    const I = re[i]*re[i]+im[i]*im[i];
    const ph = p.gamma*I;
    const c=Math.cos(ph), s=Math.sin(ph);
    const nr = re[i]*c - im[i]*s, ni = re[i]*s + im[i]*c;
    re[i]=nr; im[i]=ni;
  }

  // saturable absorber
  if(p.tauA <= 1e-9){
    for(let i=0;i<Nt;i++){
      const I = re[i]*re[i]+im[i]*im[i];
      const qi = p.q0/(1+I/p.EsatA);
      const s = Math.sqrt(Math.max(1-qi,0));
      re[i]*=s; im[i]*=s;
    }
  } else {
    for(let i=0;i<Nt;i++){
      const I = re[i]*re[i]+im[i]*im[i];
      const dq = ( -(q[i]-p.q0)/p.tauA - q[i]*I/p.EsatA );
      q[i] = Math.max(q[i]+dq, 0);
      const s = Math.sqrt(Math.max(1-q[i],0));
      re[i]*=s; im[i]*=s;
    }
  }

  // persistent noise floor
  for(let i=0;i<Nt;i++){ re[i]+=p.noise*mlRandn(); im[i]+=p.noise*mlRandn(); }

  // gain + gain-bandwidth filtering + dispersion, frequency domain
  const gSat = p.g0/(1+E/p.Esat);
  fft(re, im, false);
  for(let k=0;k<Nt;k++){
    const w = ML_omega[k];
    const amp_ = Math.exp(gSat - p.l - p.Dg*w*w);
    const ph = -0.5*p.D*w*w;
    const c=amp_*Math.cos(ph), s=amp_*Math.sin(ph);
    const nr = re[k]*c - im[k]*s, ni = re[k]*s + im[k]*c;
    re[k]=nr; im[k]=ni;
  }
  fft(re, im, true);
  return E;
}

function mlAnalyze(re, im){
  const Nt = ML_Nt;
  let peak=0, peakIdx=0, E=0; const I = new Float64Array(Nt);
  for(let i=0;i<Nt;i++){ I[i]=re[i]*re[i]+im[i]*im[i]; E+=I[i]; if(I[i]>peak){peak=I[i];peakIdx=i;} }
  // Same periodic-boundary issue as the comb-synthesis tab: the pulse can end up
  // anywhere on this cyclic grid, including right at the edge. Roll to center
  // the peak before measuring FWHM (and hand back the rolled trace for display).
  const shift = Math.floor(Nt/2) - peakIdx;
  const Irot = new Float64Array(Nt);
  for(let i=0;i<Nt;i++) Irot[((i+shift)%Nt+Nt)%Nt] = I[i];
  const peakRot = Math.floor(Nt/2);
  let lo=null, hi=null;
  for(let i=1;i<Nt;i++){
    if(Irot[i-1]<peak/2 && Irot[i]>=peak/2 && lo===null) lo=i;
    if(Irot[i-1]>=peak/2 && Irot[i]<peak/2 && i>peakRot && hi===null) hi=i;
  }
  return { I, Irot, peak, E, fwhm:(lo!==null&&hi!==null)?(hi-lo):NaN };
}

const MECH_PRESETS = {
  klm:  { tauA:0,  label:'Ideal fast SA (KLM-like)' },
  sesam:{ tauA:25, label:'Realistic SESAM' }
};
module.exports = { ML_Nt, mlRandn, fft, mlStep, mlAnalyze };
