#-------------------------------------------------------------------------------
function objectivemultigp(param; K = K, T = T, Y = Y, S = S, JITTER = JITTER)
#-------------------------------------------------------------------------------

    μ, u, w = unpack(param, K)

    objectivemultigp(; μ = μ, u = u, w = w, T = T, Y = Y, S = S, JITTER = JITTER)

end


#-------------------------------------------------------------------------------
function objectivemultigp(; μ = μ, u = u, w = w, T = T, Y = Y, S = S, JITTER = JITTER)
#-------------------------------------------------------------------------------

      N = length(T); @assert(N == length(Y) == length(S))

      F = zero(eltype(μ[1]))

      for n in 1:N

          Cₙ = sm(T[n]; w = w[n], μ = μ, u = u)

          CΣobsₙ = Cₙ + Diagonal(S[n].^2) + JITTER * I

          makematrixsymmetric!(CΣobsₙ)

          F += logpdf(MvNormal(zeros(length(T[n])), CΣobsₙ), Y[n])

      end

      return F

end


#-------------------------------------------------------------------------------
function unpack(param, K::Int)
#-------------------------------------------------------------------------------

    μ   =           param[0*K+1:K]

    u   = softplus.(param[1*K+1:2*K])

    aux = softplus.(param[2*K+1:end])

    w = collect(Iterators.partition(aux, K))

    return μ, u, w

end
