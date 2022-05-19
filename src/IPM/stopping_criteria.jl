function default_criterion(relgap, tol_relgap, φ, tol_φ)
    return (relgap > tol_relgap) || (φ > tol_φ)
end

function DIMACS_criterion(prb::InnerModel{T}, ws::Workspace{T}, tol) where {T}
    err1 = norm(prb.data.A'*prb.X - prb.data.b)/(one(T)+maximum(abs.(prb.data.b)))
    err2 = norm(prb.data.A*prb.y + prb.S - prb.data.C)/(one(T)+maximum(abs.(prb.data.C)))
    err3 = zero(T)
    err4 = zero(T)
    for (j,block) in prb.SDP_blocks
        @views smat!(ws.U[j], prb.X[block])
        symmetrize!(ws.U[j])
        λp = my_eigmin(ws.U[j])
        @views smat!(ws.U[j], prb.S[block])
        symmetrize!(ws.U[j])
        λd = my_eigmin(ws.U[j])
        if λp < err3
            err3 = λp
        end
        if λd < err4
            err4 = λd
        end
    end
    for (j,block) in prb.LP_blocks
        λp = minimum(prb.X[block])
        λd = minimum(prb.S[block])
        if λp < err3
            err3 = λp
        end
        if λd < err4
            err4 = λd
        end
    end
    err3 = max(zero(T), -err3)
    err4 = max(zero(T), -err4)
    err5 = max(zero(T), dot(prb.data.C, prb.X)-dot(prb.data.b, prb.y))
    # println("err1: $(err1), err2: $(err2), err3: $(err3), err4: $(err4), err5: $(err5)")
    @printf("err1: %.2e, err2: %.2e, err3: %.2e, err4: %.2e, err5: %.2e\n", err1, err2, err3, err4, err5)
    return (err1 > tol) ||
           (err2 > tol) ||
           (err3 > tol) ||
           (err4 > tol) ||
           (err5 > tol)
end