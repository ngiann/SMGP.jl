module SMGP

    using StatsFuns, LinearAlgebra, Random, Optim, Printf, MiscUtil, Distributions

    include("smkernel.jl")

    include("gp.jl")

    include("gpmulti2.jl"); include("objectivemultigp.jl")

    include("sincpattern.jl")

    include("sinpatterns.jl")

    export sincpatterns, sinpatterns, gp, gpmulti2

end
