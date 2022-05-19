module SMGP

    using StatsFuns, LinearAlgebra, Random, Optim, Printf, MiscUtil, Distributions

    include("smkernel.jl")

    include("gp.jl")

    include("sincpattern.jl")

    include("sinpattern.jl")

    export sincpattern, sinpattern, gp

end
