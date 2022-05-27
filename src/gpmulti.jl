function gpmulti(T, Y, S; K = K, iterations = iterations, seed = 1, initialrandom = 1, numberofrestarts = 1)

    #---------------------------------------------------------------------
    # Fix random seed for reproducibility
    #---------------------------------------------------------------------

    rg = MersenneTwister(seed)


    #---------------------------------------------------------------------
    # Set constants
    #---------------------------------------------------------------------

    JITTER = 1e-8

    N = length(T); @assert(N == length(Y) == length(S))

    numparam = K + K + N * K

    @printf("There are %d time series\n", N)

    @printf("There are %d free parameters\n", numparam)


    #---------------------------------------------------------------------
    # Auxiliary
    #---------------------------------------------------------------------

    softplus(x) = log(1 + exp(x))


    #---------------------------------------------------------------------
    function unpack(param)
    #---------------------------------------------------------------------

        @assert(length(param) == numparam)

        local μ = param[0*K+1:K]

        local u = softplus.(param[1*K+1:2*K])

        local aux = softplus.(param[2*K+1:end])

        local w = collect(Iterators.partition(aux, K))

        @assert(length(w) == N)

        return μ, u, w

    end


    #---------------------------------------------------------------------
    # Define objective as marginal log-likelihood
    #---------------------------------------------------------------------

    function objective(param)

        local μ, u, w = unpack(param)

        local F = zero(eltype(param))

        for n in 1:N

            local Cₙ = sm(T[n]; w = w[n], μ = μ, u = u)

            local CΣobsₙ = Cₙ + Diagonal(S[n].^2) + JITTER * I

            makematrixsymmetric!(CΣobsₙ)

            F += logpdf(MvNormal(zeros(length(T[n])), CΣobsₙ), Y[n])

        end

        return F

    end


    # Auxiliaries

    negativeobjective(x) = - objective(x)

    safenegativeobj = safewrapper(negativeobjective)

    safeobj = safewrapper(objective)


    #---------------------------------------------------------------------
    # Call optimiser and initialise with random search
    #---------------------------------------------------------------------

    function getsolution()

        randomsolutions = [randn(rg, numparam) for i in 1:initialrandom]

        bestindex = argmin(map(safeobj, randomsolutions))

        opt = Optim.Options(show_trace = true, iterations = iterations, show_every = 500, g_tol=1e-9)

        optimize(safenegativeobj, randomsolutions[bestindex], NelderMead(), opt)

    end


    allresults = [getsolution() for _ in 1:numberofrestarts]

    result     = allresults[argmin([res.minimum for res in allresults])]

    paramopt   = result.minimizer

    @show result.minimum

    #---------------------------------------------------------------------
    # instantiate learned matrix and observed variance parameter
    #---------------------------------------------------------------------

    μ, u, w = unpack(paramopt)

    #---------------------------------------------------------------------
    # Functions for predicting on test data
    #---------------------------------------------------------------------

    #---------------------------------------------------------------------------
    function predictTest(n, ttest)
    #---------------------------------------------------------------------------

        @assert(n > 0); @assert(n < N+1)


        Cₙ = sm(T[n]; w = w[n], μ = μ, u = u)

        CΣobsₙ = Cₙ + Diagonal(S[n].^2) + JITTER * I

        makematrixsymmetric!(CΣobsₙ)



        # dimensions: N × Ntest
        C✴ = sm(T[n], ttest; w = w[n], μ = μ, u = u)

        # Ntest × Ntest
        cB = sm(ttest;  w = w[n], μ = μ, u = u)

        # full predictive covariance
        Σpred = cB - C✴' * (CΣobsₙ \ C✴)

        makematrixsymmetric!(Σpred)

        Σpred = Σpred + JITTER*I

        # predictive mean

        μpred = C✴' * (CΣobsₙ \ Y[n])

        return μpred, Σpred

    end




    # #---------------------------------------------------------------------------
    # function predictTest(ttest::Vector{T},
    #                      ytest::Vector{T},
    #                      σtest::Vector{T}) where T<:Real
    # #---------------------------------------------------------------------------
    #
    #     local μpred, Σpred = predictTest(ttest)
    #
    #     local Σobs✴ = Diagonal(reduce(vcat, σtest).^2)
    #
    #     Σpred = Σpred + Σobs✴
    #
    #     makematrixsymmetric!(Σpred)
    #
    #     try
    #
    #         return logpdf(MvNormal(μpred, Σpred), ytest)
    #
    #     catch exception
    #
    #         if isa(exception, PosDefException)
    #
    #             local newΣpred = nearestposdef(Σpred; minimumeigenvalue = 1e-6)
    #
    #             return logpdf(MvNormal(μpred, newΣpred), ytest)
    #
    #         else
    #
    #             throw(exception)
    #
    #         end
    #
    #     end
    #
    # end


    # return:
    # • function value returned from optimisation
    # • prediction function
    # • sm parameters

    result.minimum, predictTest, (μ, u, w)

end
