// Bootstrap the exact mean of individual-seed RMSEs, not pooled MSE.
// AI-assisted implementation; no network, model fitting, or random generator.
#include <cmath>
#include <limits>
#include <vector>

extern "C" void bootstrap_primary_rmse(
    const double *losses, const int *indices, int reps, int channels, int seeds,
    int n, int lags, double practical_factor, const double *observed_theta,
    double *theta_out, double *epa_out, double *practical_out, double *log_out) {
  const double inf = std::numeric_limits<double>::infinity();
  const double eps = std::numeric_limits<double>::epsilon();
  std::vector<double> rho(channels*seeds), theta(channels), psi(channels*n);
  std::vector<double> variance(channels), cross(channels), sum(channels);
  for (int b=0; b<reps; ++b) {
    const int *ix=indices+b*n;
    bool positive=true;
    for (int j=0; j<channels; ++j) {
      theta[j]=0.;
      for (int r=0; r<seeds; ++r) {
        const double *v=losses+(j*seeds+r)*n;
        double mu=0.;
        for (int t=0; t<n; ++t) mu+=v[ix[t]];
        mu/=n;
        rho[j*seeds+r]=std::sqrt(mu);
        positive=positive && mu>0. && std::isfinite(mu);
        theta[j]+=rho[j*seeds+r]/seeds;
      }
      theta_out[b*channels+j]=theta[j];
    }
    if (!positive) {
      for (int j=1;j<channels;++j) {
        epa_out[b*(channels-1)+j-1]=inf;
        practical_out[b*(channels-1)+j-1]=inf;
        log_out[b*(channels-1)+j-1]=inf;
      }
      continue;
    }
    for (int j=0;j<channels;++j) {
      double avg=0.;
      for (int t=0;t<n;++t) {
        double value=0.;
        for (int r=0;r<seeds;++r)
          value+=losses[(j*seeds+r)*n+ix[t]]/(2.*seeds*rho[j*seeds+r]);
        value-=theta[j]/2.;
        psi[j*n+t]=value;
        avg+=value/n;
      }
      for(int t=0;t<n;++t) psi[j*n+t]-=avg;
      variance[j]=cross[j]=sum[j]=0.;
    }
    // Bartlett HAC via all zero-padded moving sums, including boundary windows.
    for (int k=0;k<n+lags;++k) {
      for(int j=0;j<channels;++j) {
        if(k<n) sum[j]+=psi[j*n+k];
        if(k>=lags+1 && k-lags-1<n) sum[j]-=psi[j*n+k-lags-1];
      }
      for(int j=0;j<channels;++j) {
        variance[j]+=sum[j]*sum[j];
        cross[j]+=sum[j]*sum[0];
      }
    }
    double denominator=static_cast<double>(n)*(lags+1);
    for(int j=0;j<channels;++j) { variance[j]/=denominator;cross[j]/=denominator; }
    for(int j=1;j<channels;++j) {
      int pos=b*(channels-1)+j-1;
      double vd=variance[j]+variance[0]-2.*cross[j];
      double vw=practical_factor*practical_factor*variance[j]+variance[0]-2.*practical_factor*cross[j];
      double vl=variance[0]/(theta[0]*theta[0])+variance[j]/(theta[j]*theta[j])-2.*cross[j]/(theta[0]*theta[j]);
      double scale_d=variance[j]+variance[0];
      double scale_w=practical_factor*practical_factor*variance[j]+variance[0];
      double scale_l=variance[0]/(theta[0]*theta[0])+variance[j]/(theta[j]*theta[j]);
      double d=(theta[j]-theta[0])-(observed_theta[j]-observed_theta[0]);
      double w=(practical_factor*theta[j]-theta[0])-(practical_factor*observed_theta[j]-observed_theta[0]);
      double l=std::log(theta[0]/theta[j])-std::log(observed_theta[0]/observed_theta[j]);
      epa_out[pos]=(vd>eps*scale_d*64.) ? d/std::sqrt(vd/n) : inf;
      practical_out[pos]=(vw>eps*scale_w*64.) ? w/std::sqrt(vw/n) : inf;
      log_out[pos]=(vl>eps*scale_l*64.) ? l/std::sqrt(vl/n) : inf;
    }
  }
}
