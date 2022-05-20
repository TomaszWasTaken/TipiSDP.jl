using MySDPSolver

using Quadmath
using DoubleFloats

dir = "MySDPSolver/problems/SDPLIB-master/data/"
files = readdir(dir, sort=false)
filter!(s -> occursin(r"arch", s), files)

TestFloats = [Float64, Double64, Float128]

for T in TestFloats
    for file in files
        prb = MySDPSolver.read_from_file(dir*file, T)
        MySDPSolver.optimize!(prb)
    end
end

nothing

# using JuMP, MathOptInterface
# using MySOCPSolver
# using LinearAlgebra

# u0 = ones(Float64, 10)
# p = ones(Float64, 10)
# q = 1.0
# model = Model()
# @variable(model, u[1:10])
# @variable(model, t)
# @objective(model, Min, t)
# @constraint(model, c1, [t, (u - u0)...] in SecondOrderCone())  # see 'splatting' for the '...' syntax
# @constraint(model, c2, u'*p == q)
# set_optimizer(model, MySOCPSolver.Optimizer)
# optimize!(model)
# println("#########################")

# using MySDPSolver
# using JuMP, MathOptInterface, Dualization
# using LinearAlgebra
# using DataFrames, CSV, Statistics

# function test_problem(name::String; N=1)
#     # Number of runs and storage
#     times_ = Vector{Float64}(undef, N)
#     iters_ = zero(Int64)
#     m_prb = zero(Int64)
#     n_prb = zero(Int64)
#     # Main loop
#     for k in 1:N
#         model = dualize(JuMP.read_from_file("problems/SDPLIB-master/data/$name.dat-s"))
#         set_optimizer(model, MySDPSolver.Optimizer)
#         optimize!(model)
#         times_[k] = MOI.get(model, MOI.SolveTime())
#         if k == N
#             iters_ = MOI.get(model, MOI.BarrierIterations())
#             m_prb = length(model.moi_backend.optimizer.model.optimizer.b)
#             n_prb = sum(abs.(model.moi_backend.optimizer.model.optimizer.blockdims))
#         end
#     end

#     return m_prb, n_prb, iters_, round(mean(times_), digits=3)
# end

# function benchmark_family(names_::Vector{String})
#     m_ = Vector{Int64}(undef, length(names_))
#     n_ = Vector{Int64}(undef, length(names_))
#     n_iters_ = Vector{Int64}(undef, length(names_))
#     t_ = Vector{Float64}(undef, length(names_))

#     for (k, name_) in enumerate(names_)
#         m_[k], n_[k], n_iters_[k], t_[k] = test_problem(name_)
#     end

#     df = DataFrame(name = names_, m = m_, n = n_, iters = n_iters_, t = t_)
#     CSV.write("benchmark_results/res_"*(replace(names_[1], Regex("[1234567890-]") => ""))*".csv", df)
# end

# test_problem("arch0")
# # benchmark_family(["arch0", "arch2", "arch4", "arch8"])
# # benchmark_family(["control1", "control2", "control3",
# #                   "control4", "control5", "control6",
# #                   "control7", "control8", "control9",
# #                   "control10", "control11"])
# # benchmark_family(["gpp100",
# #                   "gpp124-1", "gpp124-2", "gpp124-3", "gpp124-4",
# #                   "gpp250-1", "gpp250-2", "gpp250-3", "gpp250-4",
# #                   "gpp500-1", "gpp500-2", "gpp500-3", "gpp500-4"])
# # benchmark_family(["hinf1", "hinf2", "hinf3", "hinf4", "hinf5",
# #                   "hinf6", "hinf7", "hinf8", "hinf10", "hinf11",
# #                   "hinf12", "hinf13", "hinf14", "hinf15"])