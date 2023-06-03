module NewB
using LinearAlgebra
using Distributions
export AbstractMixtureModel, AbstractMvNormal, Arcsine, ArrayLikeVariate, Bernoulli, BernoulliLogit, Beta, BetaBinomial, BetaPrime, Binomial, Biweight, Categorical, Cauchy, Chernoff, Chi, Chisq, CholeskyVariate, Continuous, ContinuousDistribution, ContinuousMatrixDistribution, ContinuousMultivariateDistribution, ContinuousUnivariateDistribution, Cosine, DiagNormal, DiagNormalCanon, Dirac, Dirichlet, DirichletMultinomial, Discrete, DiscreteDistribution, DiscreteMatrixDistribution, DiscreteMultivariateDistribution, DiscreteNonParametric, DiscreteUniform, DiscreteUnivariateDistribution, Distribution, Distributions, DoubleExponential, EdgeworthMean, EdgeworthSum, EdgeworthZ, Epanechnikov, Erlang, Estimator, Exponential, FDist, FisherNoncentralHypergeometric, Frechet, FullNormal, FullNormalCanon, Gamma, GeneralizedExtremeValue, GeneralizedPareto, Geometric, Gumbel, Hypergeometric, InverseGamma, InverseGaussian, InverseWishart, IsoNormal, IsoNormalCanon, JohnsonSU, JointOrderStatistics, KSDist, KSOneSided, Kolmogorov, Kumaraswamy, LKJ, LKJCholesky, Laplace, Levy, Lindley, LocationScale, LogNormal, LogUniform, Logistic, LogitNormal, MLEstimator, MatrixBeta, MatrixDistribution, MatrixFDist, MatrixNormal, MatrixReshaped, MatrixTDist, Matrixvariate, MixtureModel, Multinomial, Multivariate, MultivariateDistribution, MultivariateMixture, MultivariateNormal, MvLogNormal, MvNormal, MvNormalCanon, MvNormalKnownCov, MvTDist, NegativeBinomial, NonMatrixDistribution, NoncentralBeta, NoncentralChisq, NoncentralF, NoncentralHypergeometric, NoncentralT, Normal, NormalCanon, NormalInverseGaussian, OrderStatistic, PGeneralizedGaussian, Pareto, Poisson, PoissonBinomial, Product, QQPair, Rayleigh, RealInterval, Rician, Sampleable, Semicircle, Skellam, SkewNormal, SkewedExponentialPower, Soliton, StudentizedRange, SufficientStats, SymTriangularDist, TDist, TriangularDist, Triweight, Truncated, TruncatedNormal, Uniform, Univariate, UnivariateDistribution, UnivariateGMM, UnivariateMixture, ValueSupport, VariateForm, VonMises, VonMisesFisher, WalleniusNoncentralHypergeometric, Weibull, Wishart, ZeroMeanDiagNormal, ZeroMeanDiagNormalCanon, ZeroMeanFullNormal, ZeroMeanFullNormalCanon, ZeroMeanIsoNormal, ZeroMeanIsoNormalCanon, canonform, ccdf, cdf, censored, cf, cgf, circvar, component, components, componentwise_logpdf, componentwise_pdf, concentration, convolve, cor, cov, cquantile, dim, dof, entropy, estimate, expected_logdet, failprob, fit, fit_mle, gradlogpdf, hasfinitesupport, insupport, invcov, invlogccdf, invlogcdf, invscale, isbounded, isleptokurtic, islowerbounded, ismesokurtic, isplatykurtic, isprobvec, isupperbounded, kldivergence, kurtosis, location, location!, logccdf, logcdf, logdetcov, logdiffcdf, loglikelihood, logpdf, logpdf!, mean, meandir, meanform, meanlogx, median, mgf, mode, modes, moment, ncategories, ncomponents, nsamples, ntrials, params, params!, partype, pdf, pdfsquaredL2norm, probs, probval, product_distribution, qqbuild, quantile, rate, sample, sample!, sampler, scale, scale!, shape, skewness, span, sqmahal, sqmahal!, std, stdlogx, succprob, suffstats, support, truncated, var, varlogx, wsample, wsample!
export mlag, genBeta, genSigma, gibbs

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

function genSigma(Y,X,beta_d,nu0,d0)
    nu1 = size(Y,1)+nu0
    d1  = d0 + only((Y-X*beta_d)'*(Y-X*beta_d)) 
    sig2_inv = rand(Gamma(nu1/2,2/d1),1)
    sig2_d = 1/only(sig2_inv)
    return sig2_d
end

function gibbs(Y,X,BETA0,Sigma0,sig2_d,d0,nu0,n_gibbs,burn)
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

end # module NewB
