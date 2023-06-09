module NewB
using LinearAlgebra
using Distributions
export mlag, genBeta, genSigma, gibbs, genSigma_old

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


function genBeta(X,Y,Beta_prior,Sigma_prior,sig2_d,beta_d)
    invSig = Sigma_prior^-1
    V = (invSig + sig2_d^(-1)*(X'*X))^-1
    C = cholesky(Hermitian(V)) 
    Beta1 =  V*(invSig*Beta_prior + sig2_d^(-1)*X'*Y)
    beta_d[:,1] = Beta1 + C.L*randn(5,1)
    return beta_d
end

function genSigma(Y,X,beta_d,nu0,d0)
    nu1 = size(Y,1)+nu0
    resid = similar(Y)          # changes 
    mul!(resid, X, beta_d)
    resid .= resid.- Y          # short form is resid .-= Y
    d1 = d0 + dot(resid, resid)
    #d1  = d0 + only((Y-X*beta_d)'*(Y-X*beta_d)) 
    sig2_inv = rand(Gamma(nu1/2,2/d1),1)
    sig2_d = 1/only(sig2_inv)
    return sig2_d
end

function genSigma_old(Y,X,beta_d,nu0,d0)
    nu1 = size(Y,1)+nu0
    resid = Y - X * beta_d;
    d1 = d0 + dot(resid, resid)
    #d1  = d0 + only((Y-X*beta_d)'*(Y-X*beta_d)) 
    sig2_inv = rand(Gamma(nu1/2,2/d1),1)
    sig2_d = 1/only(sig2_inv)
    return sig2_d
end

function gibbs(Y,X,BETA0,Sigma0,sig2_d,d0,nu0,n_gibbs,burn)
    beta_d = zeros(5,1)
    beta_dist = zeros(5,n_gibbs-burn)
    sigma_dist = zeros(1,n_gibbs-burn)
    for i = 1:n_gibbs
        beta_d = genBeta(X,Y,BETA0,Sigma0,sig2_d,beta_d)
        sig2_d = genSigma(Y,X,beta_d,nu0,d0)
        if i > burn
            beta_dist[:,i-burn] = beta_d
            sigma_dist[1,i-burn] = sig2_d
        end
    end
    return beta_dist, sigma_dist
end

end # module NewB
