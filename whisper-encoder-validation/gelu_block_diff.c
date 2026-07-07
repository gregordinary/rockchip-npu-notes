// Standalone: quantify the whisper-exact (tanh-approx GELU) vs rocket (erf GELU)
// encoder-block divergence. No NPU, no deps. Pure double-precision references of the
// SAME pre-norm Whisper encoder block, differing ONLY in the GELU variant (and we also
// test LN-variance and attention-scale conventions match). Run on x86.
//
//   block: x = x + MHA(LN1(x)); x = x + MLP(LN2(x)); MLP=GELU(h Wf1^T+bf1) Wf2^T+bf2
//   whisper: ggml_norm = (x-mean)/sqrt(var+eps), var = (1/N) sum (x-mean)^2 (BIASED)
//            GELU = 0.5 x (1+tanh(0.79788456*x*(1+0.044715 x^2)))   [ggml_gelu_f32]
//            attn scale = 1/sqrt(d_head) on scores; K has no bias
//   rocket : same, but GELU = 0.5 x (1+erf(x/sqrt2))               [rocket_encoder.c gelu_d]
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static double gelu_tanh(double x){ const double c=0.79788456080286535588, a=0.044715;
    return 0.5*x*(1.0+tanh(c*x*(1.0+a*x*x))); }
static double gelu_erf(double x){ return 0.5*x*(1.0+erf(x*M_SQRT1_2)); }

// LayerNorm row: biased variance, eps, affine (gamma,beta)
static void layernorm(const double*x,const double*g,const double*b,double eps,int T,int d,double*out){
    for(int t=0;t<T;t++){const double*xr=x+(size_t)t*d; double*o=out+(size_t)t*d;
        double m=0; for(int i=0;i<d;i++) m+=xr[i]; m/=d;
        double v=0; for(int i=0;i<d;i++){double q=xr[i]-m; v+=q*q;} v/=d;
        double r=1.0/sqrt(v+eps);
        for(int i=0;i<d;i++) o[i]=(xr[i]-m)*r*g[i]+(b?b[i]:0.0);
    }
}
// C[T,N] = X[T,K] * W[N,K]^T + bias[N]   (W row-major [out,in] = nn.Linear)
static void linear(const double*X,const double*W,const double*bias,int T,int K,int N,double*C){
    for(int t=0;t<T;t++)for(int n=0;n<N;n++){double a=bias?bias[n]:0.0;
        const double*xr=X+(size_t)t*K,*wr=W+(size_t)n*K;
        for(int k=0;k<K;k++) a+=xr[k]*wr[k]; C[(size_t)t*N+n]=a;}
}
// multi-head self-attention (whisper convention)
static void mha(const double*x,int T,int d,int nh,
                const double*Wq,const double*bq,const double*Wk,const double*Wv,const double*bv,
                const double*Wo,const double*bo,double*out){
    int dh=d/nh; double scale=1.0/sqrt((double)dh);
    double*Q=malloc((size_t)T*d*sizeof(double)),*K=malloc((size_t)T*d*sizeof(double));
    double*V=malloc((size_t)T*d*sizeof(double)),*ctx=malloc((size_t)T*d*sizeof(double));
    linear(x,Wq,bq,T,d,d,Q); linear(x,Wk,NULL,T,d,d,K); linear(x,Wv,bv,T,d,d,V);
    double*sc=malloc((size_t)T*sizeof(double));
    for(int h=0;h<nh;h++){int off=h*dh;
        for(int i=0;i<T;i++){
            double mx=-1e300;
            for(int j=0;j<T;j++){double s=0; for(int k=0;k<dh;k++) s+=Q[(size_t)i*d+off+k]*K[(size_t)j*d+off+k];
                s*=scale; sc[j]=s; if(s>mx)mx=s;}
            double sum=0; for(int j=0;j<T;j++){sc[j]=exp(sc[j]-mx); sum+=sc[j];}
            for(int k=0;k<dh;k++){double a=0; for(int j=0;j<T;j++) a+=sc[j]*V[(size_t)j*d+off+k];
                ctx[(size_t)i*d+off+k]=a/sum;}
        }
    }
    linear(ctx,Wo,bo,T,d,d,out);
    free(Q);free(K);free(V);free(ctx);free(sc);
}
// full pre-norm encoder block; gelu_kind 0=tanh(whisper) 1=erf(rocket)
static void block(const double*x,int T,int d,int nh,int dff,double eps,int gelu_kind,
    const double*ln1g,const double*ln1b,const double*Wq,const double*bq,const double*Wk,
    const double*Wv,const double*bv,const double*Wo,const double*bo,const double*ln2g,
    const double*ln2b,const double*Wf1,const double*bf1,const double*Wf2,const double*bf2,double*out){
    size_t Td=(size_t)T*d; double*ln=malloc(Td*sizeof(double)),*attn=malloc(Td*sizeof(double));
    double*xa=malloc(Td*sizeof(double)),*h=malloc((size_t)T*dff*sizeof(double));
    layernorm(x,ln1g,ln1b,eps,T,d,ln);
    mha(ln,T,d,nh,Wq,bq,Wk,Wv,bv,Wo,bo,attn);
    for(size_t i=0;i<Td;i++) xa[i]=x[i]+attn[i];
    layernorm(xa,ln2g,ln2b,eps,T,d,ln);
    linear(ln,Wf1,bf1,T,d,dff,h);
    for(size_t i=0;i<(size_t)T*dff;i++) h[i]=gelu_kind? gelu_erf(h[i]) : gelu_tanh(h[i]);
    linear(h,Wf2,bf2,T,dff,d,out);
    for(size_t i=0;i<Td;i++) out[i]+=xa[i];
    free(ln);free(attn);free(xa);free(h);
}
static double frand(unsigned*s){*s=*s*1103515245u+12345u; return ((*s>>9)&0x7fffff)/8388608.0*2-1;}
int main(int argc,char**argv){
    int T=argc>1?atoi(argv[1]):64, d=512, nh=8, dff=2048; double eps=1e-5;
    unsigned s=12345; (void)frand(&s);
    #define MK(name,n) double*name=malloc((size_t)(n)*sizeof(double)); for(size_t i=0;i<(size_t)(n);i++) name[i]=frand(&s)
    MK(x,(size_t)T*d); for(size_t i=0;i<(size_t)T*d;i++) x[i]*=2.0; // input scale ~2 (post-conv/pe range)
    MK(ln1g,d); MK(ln1b,d); MK(ln2g,d); MK(ln2b,d);
    for(int i=0;i<d;i++){ln1g[i]=ln1g[i]*0.2+1.0; ln2g[i]=ln2g[i]*0.2+1.0; ln1b[i]*=0.1; ln2b[i]*=0.1;}
    double sw=1.0/sqrt((double)d), sf=1.0/sqrt((double)d), sf2=1.0/sqrt((double)dff);
    MK(Wq,(size_t)d*d); MK(bq,d); MK(Wk,(size_t)d*d); MK(Wv,(size_t)d*d); MK(bv,d);
    MK(Wo,(size_t)d*d); MK(bo,d); MK(Wf1,(size_t)dff*d); MK(bf1,dff); MK(Wf2,(size_t)d*dff); MK(bf2,d);
    for(size_t i=0;i<(size_t)d*d;i++){Wq[i]*=sw;Wk[i]*=sw;Wv[i]*=sw;Wo[i]*=sw;}
    for(int i=0;i<dff;i++) bf1[i]*=0.1; for(int i=0;i<d;i++){bq[i]*=0.1;bv[i]*=0.1;bo[i]*=0.1;bf2[i]*=0.1;}
    for(size_t i=0;i<(size_t)dff*d;i++) Wf1[i]*=sf; for(size_t i=0;i<(size_t)d*dff;i++) Wf2[i]*=sf2;
    double*o_t=malloc((size_t)T*d*sizeof(double)),*o_e=malloc((size_t)T*d*sizeof(double));
    block(x,T,d,nh,dff,eps,0,ln1g,ln1b,Wq,bq,Wk,Wv,bv,Wo,bo,ln2g,ln2b,Wf1,bf1,Wf2,bf2,o_t);
    block(x,T,d,nh,dff,eps,1,ln1g,ln1b,Wq,bq,Wk,Wv,bv,Wo,bo,ln2g,ln2b,Wf1,bf1,Wf2,bf2,o_e);
    // cosine + max_abs of (tanh-GELU "whisper") vs (erf-GELU "rocket")
    double dot=0,na=0,nb=0,mx=0; for(size_t i=0;i<(size_t)T*d;i++){double a=o_t[i],b=o_e[i];
        dot+=a*b;na+=a*a;nb+=b*b; double e=fabs(a-b); if(e>mx)mx=e;}
    printf("BLOCK whisper(tanh-GELU) vs rocket(erf-GELU): T=%d d=%d dff=%d\n",T,d,dff);
    printf("  cosine = %.8f   max_abs = %.6f   (out RMS = %.4f)\n",dot/sqrt(na*nb),mx,sqrt(nb/((size_t)T*d)));
    // raw GELU divergence over the activation range
    double gmx=0,gat=0; for(double xx=-6;xx<=6;xx+=0.01){double e=fabs(gelu_tanh(xx)-gelu_erf(xx)); if(e>gmx){gmx=e;gat=xx;}}
    printf("  raw |tanh-GELU - erf-GELU| max = %.6f at x=%.2f\n",gmx,gat);
    return 0;
}
