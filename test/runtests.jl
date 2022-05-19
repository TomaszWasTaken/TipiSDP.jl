using MySDPSolver, Test

@testset verbose = true "All" begin
    @testset verbose = true "Model" begin
        include("./Model/problem_data.jl")
        include("./Model/problem_scaling.jl")
        include("./Model/inner_model.jl")
    end
    @testset verbose = true "LinearAlgebra" begin
        include("./LinearAlgebra/decompositions.jl")
        include("./LinearAlgebra/normal_equations.jl")
        include("./LinearAlgebra/scaling.jl")
        include("./LinearAlgebra/utils.jl")
        include("./LinearAlgebra/workspace.jl")
    end
    @testset verbose = true "IPM" begin
        include("./IPM/feasibility.jl")
        include("./IPM/predictor_corrector.jl")
        include("./IPM/residuals.jl")
    end
    @testset verbose = true "MOI" begin
        include("./Interface/MOI_wrapper.jl")
    end
end
nothing