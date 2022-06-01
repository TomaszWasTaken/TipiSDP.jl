mutable struct InnerModel{T}
    status::Symbol

    data::ProblemData{T}
    scaling::ProblemScaling{T}

    # Solution
    X::Vector{T}
    y::Vector{T}
    S::Vector{T}
    
    obj_constant::T

    optimization_sense::T
    SDP_blocks::Base.Iterators.Enumerate{Vector{UnitRange{Int64}}}
    LP_blocks::Base.Iterators.Enumerate{Vector{UnitRange{Int64}}}

    settings::Settings{T}

    function InnerModel{T}() where {T}
        return new(:SolverNotCalled,
                   ProblemData{T}(),
                   ProblemScaling{T}(),
                   T[],
                   T[],
                   T[],
                   zero(T),
                   one(T),
                   enumerate(UnitRange{Int64}[]),
                   enumerate(UnitRange{Int64}[]),
                   Settings{T}())
    end
end

function is_empty(prb::InnerModel{T}) where {T}
    return is_empty(prb.data)
end

function _empty!(prb::InnerModel{T}) where {T}
    prb.status = :Empty
    _empty!(prb.data)
    _empty!(prb.scaling)
    Base.empty!(prb.X)
    Base.empty!(prb.y)
    Base.empty!(prb.S)
    prb.SDP_blocks = enumerate(UnitRange{Int64}[])
    prb.LP_blocks = enumerate(UnitRange{Int64}[])
    return nothing
end

function add_lp_block!(prb::InnerModel{T}, n::Int64) where {T}
    add_lp_block!(prb.data, n)
    add_lp_block!(prb.scaling)
    append!(prb.X, zeros(T, n))
    append!(prb.S, zeros(T, n))
end

function add_sdp_block!(prb::InnerModel{T}, n::Int64) where {T}
    add_sdp_block!(prb.data, n)
    add_sdp_block!(prb.scaling)
    append!(prb.X, zeros(T, n*(n+1)÷2))
    append!(prb.S, zeros(T, n*(n+1)÷2))
end

function read_from_file(file::String, T::DataType)
    mat_to_vec_idx(i::Int, j::Int) = (i > j) ? mat_to_vec_idx(j, i) : div((j - 1) * j, 2) + i
    svec_scaling(i::Int, j::Int) = (i == j) ? 1.0 : sqrt(2.0)
    set_size(n::Int) = (n ≥ 0) ? n*(n+1)÷2 : abs(n)

    # prb_data = ProblemData{T}()
    # prb_scaling = ProblemScaling{T}()

    prb = InnerModel{T}()
    
    io = open(file, "r")

    num_constraints_read = false
    num_blocks_read = false
    num_variables_read = false
    num_blocks = nothing
    blocks = nothing
    b_read = false
    block_starts = [0]
    Ai_nzind = []
    Ai_nzval = []
    C_nzind = Int64[]
    C_nzval = T[]
    b = nothing
    m = nothing
    while !eof(io)
        line = strip(readline(io))
        if startswith(line, '"') || startswith(line, '*')
            continue
        end
        if !num_constraints_read
            num_constraints_read = true
            m = parse(Int64, split(line)[1])
            Ai_nzind = [Int64[] for _ in 1:m]
            Ai_nzval = [T[] for _ in 1:m]
            # println("m: ", m)
        elseif !num_blocks_read
            num_blocks_read = true
            num_blocks = parse(Int, split(line)[1])
            # println("nb of blocks: ", num_blocks)
        elseif !num_variables_read
            num_variables_read = true
            blocks = parse.(Int64, split(line))
            for i = 2:num_blocks
                push!(block_starts, block_starts[i-1] + set_size(blocks[i-1]))
            end
            # println("block_starts: ", block_starts)
            # println("blocks: ", blocks)
        elseif !b_read
            b_read = true
            b = parse.(T, split(line))
            # println("b, len(b): ", b, ", ", length(b))
        else
            values = split(line)
            cstr_num = parse(Int, values[1])
            block = parse(Int, values[2])
            row = parse(Int, values[3])
            col = parse(Int, values[4])
            coef = parse(T, values[5])
            if blocks[block] > 0
                k = block_starts[block] + mat_to_vec_idx(row, col)
                scal = convert(T, svec_scaling(row, col))
            else
                k = block_starts[block] + row
                scal = one(T)
            end
            if cstr_num == 0
                push!(C_nzind, k)
                push!(C_nzval, scal*coef)
            else
                push!(Ai_nzind[cstr_num], k)
                push!(Ai_nzval[cstr_num], scal*coef)
            end
        end
    end

    close(io)

    for block in blocks
        if block ≥ 0
            add_sdp_block!(prb, block)
        else
            add_lp_block!(prb, abs(block))
        end
    end

    n_vars = length(prb.data.C)
    # set_objective!(prb_data, sparsevec(C_nzind, C_nzval, n_vars, -))
    set_objective!(prb.data, sparsevec(C_nzind, C_nzval, n_vars, -))
    for i = 1:m
        # add_constraint!(prb_data, sparsevec(Ai_nzind[i], Ai_nzval[i], n_vars, -), b[i])
        add_constraint!(prb.data, sparsevec(Ai_nzind[i], Ai_nzval[i], n_vars, -), b[i])
    end
    # return prb_data
    prb.data.A *= -one(T)
    prb.data.b *= -one(T)
    prb.data.C *= -one(T)

    prb.SDP_blocks = enumerate(prb.data.block_indices[findall(x-> x≥0, prb.data.block_dims)])
    prb.LP_blocks = enumerate(prb.data.block_indices[findall(x-> x<0, prb.data.block_dims)])
    return prb
end

function initial_iterate!(prb::InnerModel{T}) where {T}
    prb.SDP_blocks = enumerate(prb.data.block_indices[findall(x-> x≥0, prb.data.block_dims)])
    prb.LP_blocks = enumerate(prb.data.block_indices[findall(x-> x<0, prb.data.block_dims)])

    function compute_ξ_η(j, prb)
        normsA = norm.(eachcol(prb.data.A[prb.data.block_indices[j],:]))
        normC = norm(prb.data.C[prb.data.block_indices[j]])
        sj = T(sqrt(abs(prb.data.block_dims[j])))
        ξ = T(max(10.0, sj*maximum((1.0.+abs.(prb.data.b))./(1.0.+normsA))))
        η = T(max(10.0, (1.0+max(maximum(normsA), normC))/sj))
        ξ = (prb.data.block_dims[j] > 0) ? sj*ξ : ξ
        η = (prb.data.block_dims[j] > 0) ? sj*η : η
        return ξ, η
    end

    function vec_Identity(n::Int64)
        if n > 0
            I = zeros(T, n*(n+1)÷2)
            for k = 1:n
                index = sum(1:k)
                I[index] = one(T)
            end
            return I
        else
            return ones(T, abs(n))
        end
    end

    for (j,block) in enumerate(prb.data.block_indices)
        ξ, η = compute_ξ_η(j, prb)
        prb.X[block] = ξ*vec_Identity(prb.data.block_dims[j])
        prb.S[block] = η*vec_Identity(prb.data.block_dims[j])
    end

    prb.y = zeros(T, prb.data.m)

    return nothing
end

function compute_scaling!(prb::InnerModel{T}) where {T}
    # for col in eachcol(prb.data.A)


    prb.scaling.normb =  T(max(1.0, norm(prb.data.b)))
    prb.scaling.normC = T(max(1.0, maximum(norm(prb.data.C[j]) for j in prb.data.block_indices)))
    prb.scaling.normsA = T.(max(1.0, sqrt(norm(prb.data.A[j,:]))) for j in prb.data.block_indices)
    return nothing
end

function scale_problem!(prb::InnerModel{T}) where {T}
    # scale!(one(T)/prb.scaling.normb, prb.data.b)
    prb.data.b .= prb.data.b/prb.scaling.normb
    @views for (j, block) in enumerate(prb.data.block_indices)
        # scale!(one(T)/(prb.scaling.normC*prb.scaling.normsA[j]), prb.data.C[block])
        prb.data.C[block] ./= (prb.scaling.normC*prb.scaling.normsA[j])
        # scale!(prb.scaling.normsA[j], prb.X[block])
        prb.X[block] .*= prb.scaling.normsA[j]
        # scale!(one(T)/(prb.scaling.normC*prb.scaling.normsA[j]), prb.S[block])
        prb.S[block] ./= (prb.scaling.normC*prb.scaling.normsA[j])
        # scale!(one(T)/prb.scaling.normsA[j], prb.data.A[block,:])
        prb.data.A[block,:] ./= prb.scaling.normsA[j]
    end

    return nothing
end

function unscale_problem!(prb::InnerModel{T}) where {T}
    # scale!(prb.scaling.normb, prb.data.b)
    prb.data.b .*= prb.scaling.normb
    prb.y .*= prb.scaling.normC
    @views for (j, block) in enumerate(prb.data.block_indices)
        # scale!(prb.scaling.normC*prb.scaling.normsA[j], prb.data.C[block])
        prb.data.C[block] .*= (prb.scaling.normC*prb.scaling.normsA[j])
        # scale!(one(T)/prb.scaling.normsA[j], prb.X[block])
        prb.X[block] .*= (prb.scaling.normb / prb.scaling.normsA[j])
        # scale!(prb.scaling.normC*prb.scaling.normsA[j], prb.S[block])
        prb.S[block] .*= (prb.scaling.normC*prb.scaling.normsA[j])
        # scale!(prb.scaling.normsA[j], prb.data.A[block,:])
        prb.data.A[block,:] .*= prb.scaling.normsA[j]
    end
    return nothing
end