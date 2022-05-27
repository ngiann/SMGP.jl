function sinpatterns(T, N, σ=0.01)

    y1(x) = sin(2π * x) + sin(6π * x) + sin(10π * x)

    y2(x) = sin(1π * x) + sin(6π * x) + sin(10π * x)


    x1 = [rand(T)*10π for n in 1:N]
    x2 = [rand(T)*10π for n in 1:N]

    σ1 = [ones(T)*σ for n in 1:N]
    σ2 = [ones(T)*σ for n in 1:N]

    yobs1 = [y1.(x) .+ randn(T).*σ for (x, σ) in zip(x1, σ1)]
    yobs2 = [y2.(x) .+ randn(T).*σ for (x, σ) in zip(x2, σ2)]

    vcat(x1, x2),
    vcat(yobs1, yobs2),
    vcat(σ1, σ2)

end
