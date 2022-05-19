function smat!(U::AbstractArray{T}, u::AbstractArray{T}) where {T}
    n = size(U,1)
    inv_sqrt = T(1.0/sqrt(2.0))

    @inline col_start(col::Int) = (col-1)*col÷2+1

    for k = 1:n
        c_start = col_start(k)
        for j = c_start:c_start+k-2
            @inbounds U[j-c_start+1,k] = inv_sqrt*u[j]
        end
        @inbounds U[k,k] = u[c_start+k-1]
    end
end

function svec!(u, U::Matrix{T}) where {T}
    N = size(U,1)
    index = 1

    for j = 1:N
        for i = 1:j
            if i == j
                @inbounds u[index] = U[i,j]
            else
                @inbounds u[index] = T(sqrt(2.0))*U[i,j]
            end
            index += 1
        end
    end
end

function svec_add!(u, U::Matrix{T}, α::T) where {T}
    N = size(U,1)
    index = 1

    for j = 1:N
        for i = 1:j
            if i == j
                @inbounds u[index] += α*U[i,j]
            else
                @inbounds u[index] += α*T(sqrt(2.0))*U[i,j]
            end
            index += 1
        end
    end
end

function add_transpose!(U::Matrix{T}) where {T}
    n = size(U,1)
    for j = 1:n
        for i = 1:j-1
            @inbounds U[i,j] += U[j,i]
            @inbounds U[i,j] *= T(0.5)
        end
    end
end

function symmetrize!(U)
    N = size(U,1)
    
    for j = 1:N
        for i = 1:j-1
            @inbounds U[j,i] = U[i,j]
        end
    end
end

function my_mul!(C::AbstractMatrix{T}, A::SparseMatrixCSC{T,BlasInt}, B::AbstractMatrix{T}) where {T <: Union{Float32,Float64}}
    MKLSparse.BLAS.cscmm!('N', one(T), "SUNF", A, B, zero(T), C)
    return nothing
end

function my_mul!(C::AbstractMatrix{T}, A::SparseMatrixCSC{T,BlasInt}, B::AbstractMatrix{T}) where {T <: Real}
    C .= Symmetric(A)*B
    return nothing
end

function my_inv!(R::AbstractMatrix{T}, U::AbstractMatrix{T}) where {T <: Union{Float32,Float64}}
    copyto!(U, R')
    LAPACK.trtri!('L','N', U)
end

function my_inv!(R::AbstractMatrix{T}, U::AbstractMatrix{T}) where {T <: Real}
    copyto!(U, Matrix{T}(I, size(U)))
    ldiv!(UpperTriangular(R)', U)
end