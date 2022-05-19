function sinpattern(N, σ=0.01)

    T = 1/14

    y(x) = sin(2π * x) + sin(6π * x) + sin(10π * x)

    n = collect(1:36)

    yobs = y.(n*T) + randn(N)*σ

    n*T, yobs, ones(N)*σ
end
