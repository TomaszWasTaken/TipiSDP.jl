mutable struct Workspace{T}
    # Cholesky
    L::Vector{Matrix{T}}
    R::Vector{Matrix{T}}
    # inv(S)
    invS::Vector{Matrix{T}}
    # General-purpose
    U::Vector{Matrix{T}}
    U2::Vector{Matrix{T}}
    U3::Vector{Matrix{T}}
    # Normal equations
    BB::Matrix{T}
    B::Matrix{T}
    h::Vector{T}
    hh::Matrix{T}

    H1::Vector{T}
    H2::Vector{T}

    αs::Vector{T}
    βs::Vector{T}
    τ::Vector{T}

    col_cache::Vector{Vector{BlasInt}}
    row_cache::Vector{BlasInt}
    nnz_cache::Vector{T}

    m_indices::Vector{Vector{Int64}}

    function Workspace(prb::InnerModel{T}) where {T}
        N_MAX = (any(x-> x > 0, prb.data.block_dims)) ? maximum(j*(j+1)÷2 for j in prb.data.block_dims[findall(x-> x≥0, prb.data.block_dims)]) : 1
        # if !isempty(nnz(prb.data.A[prb.data.block_indices[j], col]) for col in 1:size(prb.data.A, 2) for j in findall(x-> x≥0, prb.data.block_dims))
        NNZ_MAX = (any(x-> x > 0, prb.data.block_dims)) ? maximum(nnz(prb.data.A[prb.data.block_indices[j], col]) for col in 1:size(prb.data.A, 2) for j in findall(x-> x≥0, prb.data.block_dims)) : 1
        # else
        #     NNZ_MAX = 0
        # end
        

        m_indices = [Int64[] for j in findall(x-> x≥0, prb.data.block_dims)]
        for (j,block) in enumerate(prb.data.block_indices[findall(x-> x≥0, prb.data.block_dims)])
            for k = 1:prb.data.m
                if nnz(prb.data.A[block,k]) > 0
                    push!(m_indices[j],k)
                end
            end
        end

        return new{T}([zeros(T,prb.data.block_dims[j],prb.data.block_dims[j]) for j in findall(x-> x≥0, prb.data.block_dims)],
                      [zeros(T,prb.data.block_dims[j],prb.data.block_dims[j]) for j in findall(x-> x≥0, prb.data.block_dims)],
                      [zeros(T,prb.data.block_dims[j],prb.data.block_dims[j]) for j in findall(x-> x≥0, prb.data.block_dims)],
                      [zeros(T,prb.data.block_dims[j],prb.data.block_dims[j]) for j in findall(x-> x≥0, prb.data.block_dims)],
                      [zeros(T,prb.data.block_dims[j],prb.data.block_dims[j]) for j in findall(x-> x≥0, prb.data.block_dims)],
                      [zeros(T,prb.data.block_dims[j],prb.data.block_dims[j]) for j in findall(x-> x≥0, prb.data.block_dims)],
                      zeros(T, size(prb.data.A)),
                    #  zeros(T,prb.data.m,prb.data.m),
                      zeros(T, prb.data.m, prb.data.m),
                      zeros(T, prb.data.m),
                      zeros(T, prb.data.m, 1),
                      zeros(T, length(prb.X)),
                      zeros(T, length(prb.X)),
                      zeros(T, length(prb.data.block_dims)),
                      zeros(T, length(prb.data.block_dims)),
                      Vector{T}(undef, prb.data.m),
                      [zeros(T,prb.data.block_dims[j]+1) for j in findall(x-> x≥0, prb.data.block_dims)],
                      zeros(BlasInt, NNZ_MAX),
                      zeros(T, NNZ_MAX),
                      m_indices)
    end
end