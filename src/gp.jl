function gp(tobs, yobs, σobs; K = K, iterations = iterations, seed = 1, numberofrestarts = 1, initialrandom = 1)

    #---------------------------------------------------------------------
    # Fix random seed for reproducibility
    #---------------------------------------------------------------------

    rg = MersenneTwister(seed)


    #---------------------------------------------------------------------
    # Set constants
    #---------------------------------------------------------------------

    JITTER = 1e-8

    N = length(tobs); @assert(N == length(yobs) == length(σobs))

    #---------------------------------------------------------------------
    # Auxiliary matrices
    #---------------------------------------------------------------------

    Σobs = Diagonal(reduce(vcat, σobs).^2)

    softplus(x) = log(1+exp(x))

    #---------------------------------------------------------------------
    function unpack(param)
    #---------------------------------------------------------------------

        @assert(length(param) == K * 3)

        # local logw = (param[1+0K:1*K])

        local w = softplus.(param[1+0K:1*K])

        local μ = (param[1+1K:2*K])

        local u = softplus.(param[1+2K:3*K])

        return w, μ, u

    end


    #---------------------------------------------------------------------
    # Define objective as marginal log-likelihood
    #---------------------------------------------------------------------

    function objective(param)

        local w, μ, u = unpack(param)

        local C = sm(tobs; w, μ, u)

        local KΣobs = C + Σobs + JITTER * I

        makematrixsymmetric!(KΣobs)

        return logpdf(MvNormal(zeros(N), KΣobs), yobs)

    end


    # Auxiliaries

    negativeobjective(x) = - objective(x)

    safenegativeobj = safewrapper(negativeobjective)

    safeobj = safewrapper(objective)


    #---------------------------------------------------------------------
    # Call optimiser and initialise with random search
    #---------------------------------------------------------------------

    function randμ()

         randn(K)*3

     end

    function getsolution()

        randomsolutions = [[3*randn(K); randμ(); 3*randn(K)] for i in 1:initialrandom]

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

    w, μ, u = unpack(paramopt)

    C = sm(tobs; w, μ, u)

    KΣobs = C + Σobs + JITTER * I

    makematrixsymmetric!(KΣobs)


    #---------------------------------------------------------------------
    # Functions for predicting on test data
    #---------------------------------------------------------------------

    #---------------------------------------------------------------------------
    function predictTest(ttest::Vector{T}) where T<:Real
    #---------------------------------------------------------------------------

        # dimensions: N × Ntest
        K✴ = sm(tobs, ttest; w, μ, u)

        # Ntest × Ntest
        cB = sm(ttest; w, μ, u)

        # full predictive covariance
        Σpred = cB - K✴' * (KΣobs \ K✴)

        makematrixsymmetric!(Σpred)

        Σpred = Σpred + JITTER*I

        # predictive mean

        μpred = K✴' * (KΣobs \ yobs)

        return μpred, Σpred

    end




    #---------------------------------------------------------------------------
    function predictTest(ttest::Vector{T},
                         ytest::Vector{T},
                         σtest::Vector{T}) where T<:Real
    #---------------------------------------------------------------------------

        local μpred, Σpred = predictTest(ttest)

        local Σobs✴ = Diagonal(reduce(vcat, σtest).^2)

        Σpred = Σpred + Σobs✴

        makematrixsymmetric!(Σpred)

        try

            return logpdf(MvNormal(μpred, Σpred), ytest)

        catch exception

            if isa(exception, PosDefException)

                local newΣpred = nearestposdef(Σpred; minimumeigenvalue = 1e-6)

                return logpdf(MvNormal(μpred, newΣpred), ytest)

            else

                throw(exception)

            end

        end

    end


    # return:
    # • function value returned from optimisation
    # • prediction function
    # • sm parameters

    result.minimum, predictTest, (w, μ, u)

end
