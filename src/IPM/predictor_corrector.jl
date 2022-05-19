function predictor_corrector!(prb::InnerModel{T}) where {T}

    prb.status = :SolverNotCalled
    # Residuals
    rp = Vector{T}(undef, prb.data.m)
    Rd = Vector{T}(undef, length(prb.data.C))
    Rc = zeros(T, length(prb.data.C))

    # Solution
    δX = Vector{T}(undef, length(prb.X))
    δy = Vector{T}(undef, length(prb.y))
    δS = Vector{T}(undef, length(prb.S))

    # Init Workspace
    ws = Workspace(prb)

    update_rp!(prb, rp)
    update_Rd!(prb, Rd)

    N = T(sum(abs.(prb.data.block_dims)))
    μ = dot(prb.X, prb.S)/N
    relgap = dot(prb.X, prb.S)/(one(T)+max(abs(dot(prb.data.C,prb.X)),abs(dot(prb.data.b,prb.y))))
    pinfeas = norm(rp)/(one(T)+norm(prb.data.b))
    dinfeas = norm(Rd)/(one(T)+norm(prb.data.C))
    φ = max(pinfeas, dinfeas)

    g = 0.9

    # Main loop
    iter = 1
    iterMax = 50

    α = one(T)
    β = one(T)

@time begin
    while default_criterion(relgap, prb.settings.solution_relgap, φ, prb.settings.solution_φ)
    # while DIMACS_criterion(prb, ws, T(1e-7))
        compute_cholesky!(prb, ws)

        try
            compute_Sinv!(prb, ws, iter)
        catch e
            println("compute_Sinv! failed")
            break
        end

        update_rp!(prb, rp)
        update_Rd!(prb, Rd)

        μ = prb.X⋅prb.S/N
        relgap = dot(prb.X, prb.S)/(one(T)+max(abs(dot(prb.data.C,prb.X)),abs(dot(prb.data.b,prb.y))))
        pinfeas = norm(rp)/(one(T)+norm(prb.data.b))
        dinfeas = norm(Rd)/(one(T)+norm(prb.data.C))
        φ = max(pinfeas, dinfeas)
        
        assemble_normal_eqs!(prb, ws, rp, Rd)

        # Solve the system

        if relgap < φ
            println("relgaph < infeasibilities")
        end
        
        try 
            @views δy .= ws.B\ws.h
        catch e
            println("B is Singular")
            break
        end

        compute_ΔS!(δS, δy, Rd, prb)
        compute_δX!(δX, δS, prb, ws)
        # Feasibility
        try
            α, β = compute_step_length!(prb, ws, δX, δS, g)
        catch e
            println("Eigs fail")
            break
        end


        if μ > 1e-6
            if min(α,β) < 1.0/sqrt(3.0)
                exp = 1
            else
                exp = max(1.0, 3*min(α,β)^2)
            end
        else
            exp = 1
        end

        trXS = dot(prb.X, prb.S)
        trδXδS = α*dot(δX, prb.S) + β*dot(prb.X, δS) + α*β*dot(δX, δS)
        if trXS + trδXδS < 0.0
            σ = 0.8
        else
            frac = (trXS+trδXδS)/trXS
            σ = min(1.0, frac^exp)
        end
    
        # Update Rc
        update_Rc!(prb, ws, rp, Rd, Rc, δX, δS, σ, μ)

        @views δy .= ws.B\(ws.h)
        # @views ldiv!(UpperTriangular(ws.B)', ws.h)
        # @views ldiv!(δy, UpperTriangular(ws.B), ws.h)

        compute_ΔS!(δS, δy, Rd, prb)
        compute_ΔX!(δX, δS, prb, ws)

        try 
            α, β = compute_step_length!(prb, ws, δX, δS, g)
        catch e
            println("Eigs fail (II)")
            break
        end


        g = 0.9 + 0.09*min(α, β)

        prb.X .= prb.X .+ α.*δX
        prb.y .= prb.y .+ β.*δy
        prb.S .= prb.S .+ β.*δS

        pinfeas = norm(rp)/(one(T)+norm(prb.data.b))
        dinfeas = norm(Rd)/(one(T)+norm(prb.data.C))

        if !prb.settings.silent
            @printf("%2d     |  %.3e     |  %.3e    \n", iter, prb.optimization_sense*prb.scaling.normb*prb.scaling.normC*dot(prb.data.C, prb.X)+prb.obj_constant, prb.optimization_sense*prb.scaling.normb*prb.scaling.normC*dot(prb.data.b, prb.y)+prb.obj_constant)
        end

        if dot(prb.data.b, prb.y)/norm(prb.data.A*prb.y + prb.S) > prb.settings.primal_infeas
            prb.status = :Primal_infeasible
            break
        end

        if -dot(prb.data.C, prb.X)/norm(prb.data.A'*prb.X) > prb.settings.dual_infeas
            prb.status = :Dual_infeasible
            break
        end

        if iter == iterMax
            break
        end
        iter += 1
    end
end
    if prb.status == :SolverNotCalled
        # if @views dot(prb.data.b, prb.y)/norm(prb.data.A*prb.y + prb.S) > T(1e8)
        #     prb.status = :Primal_infeasible
        # elseif @views -dot(prb.data.C, prb.X)/norm(prb.data.A'*prb.X) > T(1e8)
        #     prb.status = :Dual_infeasible
        if iter == iterMax
            prb.status = :Maximum_number_of_iterations_reached
        else
            prb.status = :Solved
        end
    end
    if !prb.settings.silent
        println("Primal obj: ", prb.optimization_sense*prb.scaling.normb*prb.scaling.normC*dot(prb.data.C, prb.X))
        println("Dual obj:   ", prb.optimization_sense*prb.scaling.normb*prb.scaling.normC*dot(prb.data.b, prb.y))
        println("X⋅S: ", dot(prb.X, prb.S))
        println("relgap: ", relgap)
        println("φ: ", φ)
    end
end

function optimize!(prb::InnerModel{T}) where {T}
    # println("1: ", prb.data.A)sss
    compute_scaling!(prb)
    # println("2: ", prb.data.A)
    initial_iterate!(prb)
    # println("3: ", prb.data.A)
    scale_problem!(prb)
    println(prb.scaling)
    # println("4: ", prb.data.A)
    predictor_corrector!(prb)
    unscale_problem!(prb)
    return nothing
end
