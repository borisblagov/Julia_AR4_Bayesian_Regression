module NewB
using LinearAlgebra
using Distributions
export mlag, genBeta, genSigma, gibbs, genBeta!, genSigma!

"""
    mlag(Yfull::Matrix{Float64},p::Integer)
    Creates lags of a matrix for a VAR representation with a constant on the left
"""
function mlag(Yfull::Matrix{Float64},p::Integer)
    (Tf, n) = size(Yfull)
    X = ones(Tf-p,1)
    for i = 1:p
        X = [X Yfull[p-i+1:end-i,:]]
    end
    Y = Yfull[p+1:end,:]
    return X, Y             # this changes the array passed into the function
end


function genBeta(X,Y,Beta_prior,Sigma_prior,sig2_d)
    invSig = Sigma_prior^-1
    V = (invSig + sig2_d^(-1)*(X'*X))^-1
    C = cholesky(Hermitian(V)) 
    Beta1 =  V*(invSig*Beta_prior + sig2_d^(-1)*X'*Y)
    beta_d = Beta1 + C.L*randn(5,1)
    return beta_d
end


function genBeta!(beta_d,X,Y,Beta_prior,Sigma_prior,sig2_d_vec)
    invSig = Sigma_prior^-1
    V = (invSig + sig2_d_vec[1]^(-1)*(X'*X))^-1
    C = cholesky(Hermitian(V)) 
    Beta1 =  V*(invSig*Beta_prior + sig2_d_vec[1]^(-1)*X'*Y)
    beta_d .= Beta1 .+ C.L*randn(5,1)
end

function genSigma(Y,X,beta_d,nu0,d0)
    nu1 = size(Y,1)+nu0
    resid = Y - X * beta_d;
    d1 = d0 + dot(resid, resid)
    #d1  = d0 + only((Y-X*beta_d)'*(Y-X*beta_d)) 
    sig2_inv = rand(Gamma(nu1/2,2/d1),1)
    sig2_d = 1/only(sig2_inv)
    return sig2_d
end


function genSigma!(sig2_d_vec,Y,X,beta_d,nu0,d0)
    nu1 = size(Y,1)+nu0
    resid = Y - X * beta_d;
    d1 = d0 + dot(resid, resid)
    #d1  = d0 + only((Y-X*beta_d)'*(Y-X*beta_d)) 
    sig2_inv = rand(Gamma(nu1/2,2/d1),1)
    sig2_d_vec[:] = 1.0 ./sig2_inv
end

function gibbs_old(Y,X,BETA0,Sigma0,sig2_d,d0,nu0,n_gibbs,burn)
    beta_dist = zeros(5,n_gibbs-burn)
    sigma_dist = zeros(1,n_gibbs-burn)
    for i = 1:n_gibbs
        beta_d = genBeta(X,Y,BETA0,Sigma0,sig2_d)
        sig2_d = genSigma(Y,X,beta_d,nu0,d0)
        if i > burn
            beta_dist[:,i-burn] = beta_d
            sigma_dist[1,i-burn] = sig2_d
        end
    end
    return beta_dist, sigma_dist
end


function gibbs(Y,X,BETA0,Sigma0,sig2_d_init,d0,nu0,n_gibbs,burn)
    beta_d = similar(BETA0)
    sig2_d_vec = sig2_d_init
    beta_dist = zeros(5,n_gibbs-burn)
    sigma_dist = zeros(1,n_gibbs-burn)
    for i = 1:n_gibbs
        genBeta!(beta_d,X,Y,BETA0,Sigma0,sig2_d_vec)
        genSigma!(sig2_d_vec,Y,X,beta_d,nu0,d0)
        if i > burn
            beta_dist[:,i-burn] .= beta_d[:]
            sigma_dist[:,i-burn] .= sig2_d_vec[:]
        end
    end
    return beta_dist, sigma_dist
end

end # module NewB
