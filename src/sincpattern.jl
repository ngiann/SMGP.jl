function sincpattern(N,σ=0.01)

    y(x) = sinc(x+10) + sinc(x) + sinc(x-10)

    x = rand(Uniform(-12,12), N)

    yobs = y.(x) + randn(N)*σ

    x, yobs, ones(N)*σ
end
