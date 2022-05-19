# using LinearAlgebra, SparseArrays

# import LinearAlgebra: BlasInt

mutable struct ProblemData{T}
    # structure
    m::Int64
    block_dims::Vector{Int64}
    block_indices::Vector{UnitRange{Int64}}

    # problem data
    A::SparseMatrixCSC{T, BlasInt}
    b::Vector{T}
    C::Vector{T}

    p::Vector{BlasInt}

    function ProblemData{T}() where {T}
            return new(zero(Int64),       # m
                       Int64[],           # block_dims
                       UnitRange{Int64}[], 
                       spzeros(T, 0, 0),  # A
                       T[],               # b
                       T[],
                       BlasInt[])               # C
    end
end

function is_empty(prb::ProblemData{T}) where {T}
    return prb.m == 0 &&
           isempty(prb.block_dims) &&
           isempty(prb.block_indices) &&
           prb.A == spzeros(T, 0, 0) &&
           isempty(prb.b) &&
           isempty(prb.C)
 end

# import Base.empty!

function _empty!(prb::ProblemData{T}) where {T}
    prb.m = zero(Int64)
    Base.empty!(prb.block_dims)
    Base.empty!(prb.block_indices)
    prb.A = spzeros(T, 0, 0)
    Base.empty!(prb.b)
    Base.empty!(prb.C)
    return nothing
end

function add_lp_block!(prb::ProblemData{T}, n::Int64) where {T}
    push!(prb.block_dims, -n)
    if isempty(prb.block_indices)
        push!(prb.block_indices, 1:n)
    else
        push!(prb.block_indices, prb.block_indices[end].stop+1:prb.block_indices[end].stop+n)
    end
    prb.A = vcat(prb.A, spzeros(T, n, prb.m))
    prb.C = vcat(prb.C, zeros(T, n))
    return nothing
end

function add_sdp_block!(prb::ProblemData{T}, n::Int64) where {T}
    push!(prb.block_dims, n)
    if isempty(prb.block_indices)
        push!(prb.block_indices, 1:n*(n+1)÷2)
    else
        push!(prb.block_indices, prb.block_indices[end].stop+1:prb.block_indices[end].stop+(n*(n+1)÷2))
    end
    prb.A = vcat(prb.A, spzeros(T, n*(n+1)÷2, prb.m))
    prb.C = vcat(prb.C, zeros(T, n*(n+1)÷2))
    return nothing
end

function set_objective!(prb::ProblemData{T}, obj) where {T}
    # copy!(prb.C, obj)
    for i = 1:length(prb.C)
        prb.C[i] = obj[i]
    end
    return nothing
end

function add_constraint!(prb::ProblemData{T}, cstr, RHS::T) where {T}
    prb.m += 1
    prb.A = hcat(prb.A, cstr)
    push!(prb.b, RHS)
    return nothing
end

function sort_A!(prb::ProblemData{T}) where {T}
    prb.p = sortperm([nnz(col) for col in eachcol(prb.A)], rev=true)
    permute!(prb.A, 1:size(prb.A,1), prb.p)
    # dropzeros!(prb.A)
    permute!(prb.b, prb.p)
    return nothing
end

function invsort_A!(prb::ProblemData{T}) where {T}
    permute!(prb.A, 1:size(prb.A,1), invperm(prb.p))
    invpermute!(prb.b, prb.p)
    return nothing
end
