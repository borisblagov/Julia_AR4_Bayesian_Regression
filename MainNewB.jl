using Revise
using LinearAlgebra
using Distributions
using DelimitedFiles
using BenchmarkTools
using NewB
# ENV["GTK_AUTO_IDLE"] = false
# import ProfileView

 fdata = readdlm("gdp4795.txt")
 Z = 100*fdata[21:end,:]./fdata[20:end-1,:].-100
# plot(Z)
#Z = rand(150,1)

n_gibbs = 7000
burn    = 2000
p = 4;

(X,Y) = mlag(Z,p)

Sigma0   = I(5)*4000
BETA0   = zeros(5,1) 
sig2_d    = 0.5        
sig2_d_init    = [0.5]        
nu0 =0
d0  =0



#beta_d = genBeta(X,Y,BETA0,Sigma0,sig2_d)
#sig2_d = genSigma(Y,X,beta_d,nu0,d0)

(beta_dist, sigma_dist) = gibbs(Y,X,BETA0,Sigma0,sig2_d_init,d0,nu0,n_gibbs,burn)
#(beta_dist, sigma_dist) = gibbs_old3(Y,X,BETA0,Sigma0,sig2_d_init,d0,nu0,n_gibbs,burn)

mean(beta_dist,dims=2)
mean(sigma_dist,dims=2)