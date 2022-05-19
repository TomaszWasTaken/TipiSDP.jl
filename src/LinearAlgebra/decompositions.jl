function compute_cholesky!(prb::InnerModel{T}, ws::Workspace{T}) where {T}
    for (j,block) in prb.SDP_blocks
        @views smat!(ws.L[j], prb.X[block])
        @views smat!(ws.R[j], prb.S[block])

        @views my_chol!(ws.L[j])
        @views my_chol!(ws.R[j])
    end
    return nothing
end

function invR!(R::AbstractMatrix{T}, U::AbstractMatrix{T}) where {T <: Union{Float32,Float64}}
    copyto!(U, R)
    LAPACK.trtri!('U','N', U)
end

function invR!(R::AbstractMatrix{T}, U::AbstractMatrix{T}) where {T <: Real}
    copyto!(U, Matrix{T}(I, size(U)))
    ldiv!(UpperTriangular(R), U)
end

function compute_Sinv!(prb::InnerModel{T}, ws::Workspace{T}, iter) where {T}
    for (j,_) in prb.SDP_blocks
        invR!(ws.R[j], ws.U[j])
        mul!(ws.invS[j], UpperTriangular(ws.U[j]), UpperTriangular(ws.U[j])')
        if (iter == 2) || (iter == 3)
            @views ws.invS[j] += T(1e-16)I
        end
    end

    return nothing
end

function my_chol!(U::AbstractMatrix{T}) where {T <: Union{Float32,Float64}}
    LAPACK.potrf!('U', U)
    triu!(U)
end

function my_chol!(U)
    cholesky!(Hermitian(U))
    triu!(U)
end