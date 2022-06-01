module TipiSDP

using LinearAlgebra, SparseArrays, MKLSparse, SuiteSparse
using GenericLinearAlgebra
using Printf
using MathOptInterface

import LinearAlgebra: BlasInt

import Base.empty!

function version()
    v"0.1.0"
end

include("./LinearAlgebra/utils.jl")
include("./LinearAlgebra/scaling.jl")

#TODO: order matters
include("./Model/problem_data.jl")
include("./Model/problem_scaling.jl")
include("./Model/settings.jl")
include("./Model/inner_model.jl")

include("./LinearAlgebra/workspace.jl")
include("./IPM/residuals.jl")
include("./IPM/feasibility.jl")
include("./IPM/stopping_criteria.jl")
include("./IPM/predictor_corrector.jl")

include("./LinearAlgebra/decompositions.jl")
include("./LinearAlgebra/normal_equations.jl")

include("./Interface/MOI_wrapper.jl")
end # module
