# Residuals
function update_rp!(prb::InnerModel{T}, rp::Vector{T}) where {T <: Union{Float32,Float64}}
    BLAS.blascopy!(prb.data.m, prb.data.b, 1, rp, 1)
    MKLSparse.BLAS.cscmv!('T', -one(T), "GU2F", prb.data.A, prb.X, one(T), rp)
    return nothing
end

function update_rp!(prb::InnerModel{T}, rp::Vector{T}) where {T <: Real}
    @views rp .= prb.data.b - prb.data.A'*prb.X
    return nothing
end

function update_Rd!(prb::InnerModel{T}, Rd::Vector{T}) where {T <: Union{Float32,Float64}}
    BLAS.blascopy!(length(Rd), prb.data.C, 1, Rd, 1)
    BLAS.axpy!(-one(T), prb.S, Rd)
    MKLSparse.BLAS.cscmv!('N', -one(T), "GU2F", prb.data.A, prb.y, one(T), Rd)
    return nothing
end

function update_Rd!(prb::InnerModel{T}, Rd::Vector{T}) where {T <: Real}
    # copy!(Rd, prb.data.C)
    @views Rd .= prb.data.C - prb.S - prb.data.A*prb.y
    return nothing
end

function update_Rc!(prb::InnerModel{T}, ws::Workspace{T}, rp, Rd, Rc::Vector{T}, δX, δS, σ, μ) where {T}
    for (j,block) in prb.SDP_blocks
        # @views my_inv!(ws.R[j], ws.U3[j])
        # @views ldiv!(UpperTriangular(ws.R[j]), ws.U3[j])
        
        @views svec!(ws.H2[block], ws.invS[j])
        @views ws.H2[block] .*= σ*μ
        @views ws.H2[block] .-= prb.X[block]
        
        @views smat!(ws.U[j], δX[block])
        symmetrize!(ws.U[j])
        @views smat!(ws.U2[j], δS[block])
        symmetrize!(ws.U2[j])
        
        @views mul!(ws.U3[j], ws.U2[j], ws.U[j])
        # @views ldiv!(UpperTriangular(ws.R[j])', ws.U3[j])
        # @views ldiv!(UpperTriangular(ws.R[j]), ws.U3[j])
        @views mul!(ws.U2[j], ws.invS[j], ws.U3[j])
        
        add_transpose!(ws.U2[j])
        @views svec_add!(ws.H2[block], ws.U2[j], -one(T))
    end

    for (j,block) in prb.LP_blocks
        for i in block
            ws.H2[i] = σ*μ/prb.S[i] - prb.X[i] - δX[i]*δS[i]/prb.S[i]
        end
    end

    # ws.h .= rp + prb.data.A'*(ws.H1-ws.H2)
    @views ws.H1 .-= ws.H2
    @views mul!(ws.h, prb.data.A', ws.H1)
    @views ws.h .+= rp

    # @views ws.h .= rp + prb.data.A'*(ws.H1-ws.H2)
end

function compute_ΔS!(δS, δy, Rd, prb::InnerModel{T}) where {T <: Union{Float32,Float64}}
    BLAS.blascopy!(length(Rd), Rd, 1, δS, 1)
    MKLSparse.BLAS.cscmv!('N', -one(T), "GU2F", prb.data.A, δy, one(T), δS)
    return nothing
end

function compute_ΔS!(δS, δy, Rd, prb::InnerModel{T}) where {T <: Real}
    δS .= Rd - prb.data.A*δy
    return nothing
end

function compute_ΔX!(δX, δS, prb, ws)
    for (j,block) in prb.SDP_blocks
        @views smat!(ws.U[j], prb.X[block])
        symmetrize!(ws.U[j])
        @views smat!(ws.U3[j], δS[block])
        symmetrize!(ws.U3[j])
        # ws.U2[j] .= Symmetric(ws.U3[j])*Symmetric(ws.U[j])
        @views mul!(ws.U2[j], ws.U3[j], ws.U[j])
        @views ldiv!(UpperTriangular(ws.R[j])', ws.U2[j])
        @views ldiv!(UpperTriangular(ws.R[j]), ws.U2[j])
        # ws.U2[j] .= UpperTriangular(ws.R[j])'\Symmetric(ws.U2[j])
        # ws.U2[j] .= UpperTriangular(ws.R[j])\Symmetric(ws.U2[j])
        # @views BLAS.trsm!('L','U','T','N', 1.0, ws.R[j], ws.U2[j])
        # @views BLAS.trsm!('L','U','N','N', 1.0, ws.R[j], ws.U2[j])
        add_transpose!(ws.U2[j])
        @views svec!(δX[block], ws.U2[j])
    end
    for (j,block) in prb.LP_blocks
        for i in block
            @inbounds δX[i] = δS[i]*prb.X[i]/prb.S[i]
        end
    end
    @views δX .= ws.H2 .- δX
end