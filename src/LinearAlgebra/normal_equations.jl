function assemble_normal_eqs!(prb, ws, rp, Rd)

    if size(ws.BB) == size(prb.data.A)
        for (j,block) in prb.SDP_blocks
            @views smat!(ws.U[j], prb.X[block])
            @views symmetrize!(ws.U[j])

            for k in ws.m_indices[j]
                sparse_A_mul!(ws.U2[j], prb.data.A[block,k], ws.U[j], ws.col_cache[j], ws.row_cache, ws.nnz_cache)
                
                @views mul!(ws.U3[j], ws.invS[j], ws.U2[j])

                add_transpose!(ws.U3[j])

                @views svec!(ws.BB[block,k], ws.U3[j])
            end

            @views smat!(ws.U3[j], Rd[block])
            @views symmetrize!(ws.U3[j])

            @views mul!(ws.U2[j], ws.U3[j], ws.U[j])

            @views mul!(ws.U3[j], ws.invS[j], ws.U2[j])

            add_transpose!(ws.U3[j])
            @views svec!(ws.H1[block], ws.U3[j])

            @views ws.H2[block] .= -prb.X[block]
        end
        
        for (j,block) in prb.LP_blocks
            for i in block
                for k = 1:prb.data.m
                    @inbounds ws.BB[i,k] = prb.data.A[i,k]*prb.X[i]/prb.S[i]
                end
                @inbounds ws.H1[i] = Rd[i]*prb.X[i]/prb.S[i]
                @inbounds ws.H2[i] = -prb.X[i]
            end
        end

        @views mul!(ws.B, prb.data.A', ws.BB)
    else
        fill!(ws.B, zero(Float64))
        max_block = maximum(blk for blk in prb.data.block_dims)
        N = max_block*(max_block+1)÷2
        t = zeros(Float64, N, 1)
        AT = prb.data.A'
        for (j,block) in prb.SDP_blocks
            fill!(ws.BB, zero(Float64))
            @views smat!(ws.U[j], prb.X[block])
            @views symmetrize!(ws.U[j])

            for k in ws.m_indices[j]
                sparse_A_mul!(ws.U2[j], prb.data.A[block,k], ws.U[j], ws.col_cache[j], ws.row_cache, ws.nnz_cache)
                
                @views mul!(ws.U3[j], ws.invS[j], ws.U2[j])

                add_transpose!(ws.U3[j])

                @views svec!(t[1:size(prb.X[block],1)], ws.U3[j])
                # @views ws.BB[:,k] .= AT[:,block]*t[1:size(prb.X[block],1)]
                @views mul!(ws.BB[:,k], AT[:,block], t[1:size(prb.X[block],1)])
            end

            ws.B .+= ws.BB

            @views smat!(ws.U3[j], Rd[block])
            @views symmetrize!(ws.U3[j])

            @views mul!(ws.U2[j], ws.U3[j], ws.U[j])

            @views mul!(ws.U3[j], ws.invS[j], ws.U2[j])

            add_transpose!(ws.U3[j])
            @views svec!(ws.H1[block], ws.U3[j])

            @views ws.H2[block] .= -prb.X[block]
        end
        
        for (j,block) in prb.LP_blocks
            for i in block
                for k = 1:prb.data.m
                    @inbounds ws.BB[i,k] = prb.data.A[i,k]*prb.X[i]/prb.S[i]
                end
                @inbounds ws.H1[i] = Rd[i]*prb.X[i]/prb.S[i]
                @inbounds ws.H2[i] = -prb.X[i]
            end
        end
    end

    @views ws.H2 .-= ws.H1
    @views ws.H2 .*= -1.0
    @views mul!(ws.h, prb.data.A', ws.H2)
    @views ws.h .+= rp

end

function ind2cart(i)
    m = (isqrt(8*i+1)-1)÷2
    t1 = m*(m+1)÷2
    t2 = (m*m + 3*m +2)÷2
    if t1 == i
        t = t1
    else
        t = max(t1,t2)
    end
    y = isqrt(2*t)
    x = y - (t-i)
    # println(x,", ",y)
    return x, y
end


function sparse_A_mul!(C::Matrix{T}, A, B, colptr, rowval, nzval) where {T}
    n = size(B,1)
    NNZ = length(A.nzind)

    fill!(colptr, zero(BlasInt))

    colptr[1] = 1

    for k = 1:NNZ
        i,j = ind2cart(A.nzind[k])
        @inbounds colptr[j+1] += 1
        @inbounds rowval[k] = i
        @inbounds nzval[k] = i == j ? A.nzval[k] : A.nzval[k]/T(sqrt(2.0))
    end

    for k = 1:n
        @inbounds colptr[k+1] += colptr[k]
    end
    
    mat = SparseMatrixCSC{T, BlasInt}(n, n, colptr, rowval[1:NNZ], nzval[1:NNZ])

    my_mul!(C, mat, B)
end

function compute_δX!(δX, δS, prb, ws)
    for (j,block) in prb.SDP_blocks
        @views smat!(ws.U3[j], δS[block])
        @views symmetrize!(ws.U3[j])
        # ws.U2[j] .= Symmetric(ws.U3[j])*Symmetric(ws.U[j])
        @views mul!(ws.U2[j], ws.U3[j], ws.U[j])
        # ws.U2[j] .= triu(ws.R[j])'\Symmetric(ws.U2[j])
        # ws.U2[j] .= triu(ws.R[j])\Symmetric(ws.U2[j])
        # @views BLAS.trsm!('L','U','T','N', 1.0, ws.R[j], ws.U2[j])
        # @views BLAS.trsm!('L','U','N','N', 1.0, ws.R[j], ws.U2[j])
        # @views ldiv!(UpperTriangular(ws.R[j])', ws.U2[j])
        # @views ldiv!(UpperTriangular(ws.R[j]), ws.U2[j])
        @views mul!(ws.U3[j], ws.invS[j], ws.U2[j])
        add_transpose!(ws.U3[j])
        @views svec!(δX[block], ws.U3[j])
    end
    for (j,block) in prb.LP_blocks
        for i in block
            @inbounds δX[i] = δS[i]*prb.X[i]/prb.S[i]
        end
    end
    δX .*= -1.0
    δX .-= prb.X
end
