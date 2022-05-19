function sm(x::Real, x′::Real; w, μ, u)

    Q = length(w)

    @assert(Q == length(μ) == length(u))

    κ = zero(eltype(x))

    for q in 1:Q

        τ = x - x′

        κ += w[q] * exp(-2π^2 * τ^2 * u[q]) * cos(2*π*τ*μ[q]) # eq. 12

    end

    κ

end


function sm(x::Vector{T}; w, μ, u) where T<:Real

    sm(x, x; w = w, μ = μ, u = u)

end

function sm(x::Vector{T}, x′::Vector{T}; w, μ, u) where T<:Real

    K = zeros(eltype(x), length(x), length(x′))

    for i in 1:length(x), j in 1:length(x′)

        K[i,j] = sm(x[i], x′[j]; w=w, μ=μ, u=u)

    end

    K

end
