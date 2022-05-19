function compute_step_length!(prb::InnerModel{T}, ws::Workspace{T}, δX, δS, g) where {T}
    index = 1
    for (j,block) in prb.SDP_blocks
        @views smat!(ws.U[j], δX[block])
        symmetrize!(ws.U[j])

        @views rdiv!(ws.U[j], UpperTriangular(ws.L[j]))
        @views ldiv!(UpperTriangular(ws.L[j])', ws.U[j])

        λa = my_eigmin(ws.U[j])

        @views smat!(ws.U[j], δS[block])
        symmetrize!(ws.U[j])

        @views rdiv!(ws.U[j], UpperTriangular(ws.R[j]))
        @views ldiv!(UpperTriangular(ws.R[j])', ws.U[j])

        λb = my_eigmin(ws.U[j])
        
        @inbounds ws.αs[index] = (λa < 0.0) ? -1.0/λa : Inf
        @inbounds ws.βs[index] = (λb < 0.0) ? -1.0/λb : Inf
        index += 1
    end

    for (j,block) in prb.LP_blocks
        λa = Inf
        λb = Inf

        @inbounds for i in block
            if δX[i]/prb.X[i] < λa
                λa = δX[i]/prb.X[i]
            end
            if δS[i]/prb.S[i] < λb
                λb = δS[i]/prb.S[i]
            end
        end

        @inbounds ws.αs[index] = (λa < 0.0) ? -1.0/λa : Inf
        @inbounds ws.βs[index] = (λb < 0.0) ? -1.0/λb : Inf
        index += 1
    end

    α = min(1.0, g*minimum(ws.αs))     # max. step lengths
    β = min(1.0, g*minimum(ws.βs))

    return α, β
end

function my_eigmin(U::Matrix{T}) where {T <: Union{Float32,Float64}}
    λ = LAPACK.syevr!('N','I','U', U, zero(T), zero(T), 1, 1, T(1e-6))[1][1]
    return λ
end

function my_eigmin(U)
    λ = GenericLinearAlgebra._eigvals!(Hermitian(U))
    return minimum(λ)
end