// Optional acceleration of the EXACT numpy rolling-HAC bootstrap calculation.
// No RNG, model fitting, critical values, p-values, or decisions occur here.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
extern "C" void bootstrap_hac(const double* losses, const int* indices,
 int batch, int channels, int n, int lags, double a, const double* observed,
 double* means, double* td, double* tw, double* tl) {
 const double eps = std::numeric_limits<double>::epsilon()*64;
 const double tiny = std::numeric_limits<double>::min();
 const double inf = std::numeric_limits<double>::infinity();
 std::vector<double> prefix(channels*(n+1)), first(n+lags), average(channels);
 const auto studentize = [&](double num, double variance, double scale) {
   return variance > eps*std::max(scale,tiny) ? num/std::sqrt(variance/n) : inf;
 };
 for (int b=0;b<batch;++b) {
   const int* idx=indices+b*n;
   for(int j=0;j<channels;++j) {
     const double* q=losses+j*n;
     double sum=0;
     for(int t=0;t<n;++t) sum+=q[idx[t]];
     average[j]=sum/n;
     means[b*channels+j]=average[j];
     double* cs=prefix.data()+j*(n+1);cs[0]=0;
     for(int t=0;t<n;++t) cs[t+1]=cs[t]+q[idx[t]]-average[j];
   }
   const double* cs0=prefix.data();
   double vc=0;
   for(int t=0;t<n+lags;++t) {
     first[t]=cs0[std::min(t+1,n)]-cs0[std::max(t-lags,0)];
     vc+=first[t]*first[t];
   }
   vc/=n*(lags+1.0);
   for(int j=1;j<channels;++j) {
     const double* cs=prefix.data()+j*(n+1);
     double vj=0,cv=0;
     for(int t=0;t<n+lags;++t) {
       double w=cs[std::min(t+1,n)]-cs[std::max(t-lags,0)];
       vj+=w*w;cv+=w*first[t];
     }
     vj/=n*(lags+1.0);cv/=n*(lags+1.0);
     int at=b*(channels-1)+(j-1);
     double mc=average[0],mr=average[j],oc=observed[0],orr=observed[j];
     td[at]=studentize(mr-mc-(orr-oc),vj+vc-2*cv,vj+vc);
     tw[at]=studentize(a*mr-mc-(a*orr-oc),a*a*vj+vc-2*a*cv,a*a*vj+vc);
     if(mc>0 && mr>0 && oc>0 && orr>0) {
       double scale=vc/(mc*mc)+vj/(mr*mr);
       tl[at]=studentize(std::log(mc/mr)-std::log(oc/orr),scale-2*cv/(mc*mr),scale);
     } else tl[at]=inf;
   }
 }
}
