using MathOptInterface

const MOI = MathOptInterface

mutable struct Optimizer{T} <: MOI.AbstractOptimizer
    inner::InnerModel{T}
    varmap::Vector{Tuple{Int,Int,Int}}

    objective_sense::Int
    silent::Bool
    options::Dict{Symbol, Any}
    function Optimizer{T}(;kwargs...) where {T}
        opt = new{T}(
              InnerModel{T}(),
              Tuple{Int,Int,Int}[],
              0,
              false,
              Dict{Symbol, Any}()
        )

        return opt
    end
end

Optimizer(;kwargs...) = Optimizer{Float64}(;kwargs...)

varmap(opt::Optimizer, vi::MOI.VariableIndex) = opt.varmap[vi.value]

MOI.get(::Optimizer, ::MOI.SolverName) = "TipiSDP"

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(opt::Optimizer, ::MOI.Silent, val::Bool)
    opt.silent = val
end

MOI.get(opt::Optimizer, ::MOI.Silent) = opt.silent

const RAW_STATUS = "RAW_STRING_PLACEHOLDER"

function MOI.get(opt::Optimizer, ::MOI.RawStatusString)
	return RAW_STATUS
end

MOI.get(opt::Optimizer, ::MOI.SolveTimeSec) = 0.0

const SupportedSets = Union{MOI.Nonnegatives, MOI.PositiveSemidefiniteConeTriangle}

MOI.supports_add_constrained_variables(::Optimizer, ::Type{<:SupportedSets}) = true
MOI.supports_add_constrained_variables(::Optimizer, ::Type{MOI.Reals}) = false

function MOI.supports(::Optimizer{T},::Union{MOI.ObjectiveSense,MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}}) where {T}
    return true
end

function MOI.supports_constraint(
    ::Optimizer{T}, ::Type{MOI.ScalarAffineFunction{T}},
    ::Type{MOI.EqualTo{T}}) where {T}
    return true
end

function MOI.empty!(opt::Optimizer)
    _empty!(opt.inner)
    Base.empty!(opt.varmap)
    return nothing
end

function MOI.is_empty(opt::Optimizer)
    return is_empty(opt.inner)
end

function MOI.optimize!(opt::Optimizer)
    println("optimize! called")

    if isempty(opt.inner.data.b)
        println("No constraints")
        opt.inner.status = :Dual_infeasible
    else
        optimize!(opt.inner)
    end
end

function _new_block(opt::Optimizer, set::MOI.Nonnegatives)
    # push!(optimizer.blockdims, -MOI.dimension(set))
    add_lp_block!(opt.inner, MOI.dimension(set))
    blk = length(opt.inner.data.block_dims)
    for i in 1:MOI.dimension(set)
        push!(opt.varmap, (blk, i, i))
    end
end

function _new_block(opt::Optimizer, set::MOI.PositiveSemidefiniteConeTriangle)
    # push!(optimizer.blockdims, set.side_dimension)
    add_sdp_block!(opt.inner, set.side_dimension)
    blk = length(opt.inner.data.block_dims)
    for i in 1:set.side_dimension
        for j in 1:i
            push!(opt.varmap, (blk, i, j))
        end
    end
end

function _add_constrained_variables(optimizer::Optimizer,
                                             set::SupportedSets)
    offset = length(optimizer.varmap)
    _new_block(optimizer, set)
    ci = MOI.ConstraintIndex{MOI.VectorOfVariables, typeof(set)}(offset + 1)
    return [MOI.VariableIndex(i) for i in offset .+ (1:MOI.dimension(set))], ci
end

function constrain_variables_on_creation(
    dest::MOI.ModelLike,
    src::MOI.ModelLike,
    index_map::MOI.Utilities.IndexMap,
    ::Type{S},
) where {S<:MOI.AbstractVectorSet}
    for ci_src in MOI.get(src, MOI.ListOfConstraintIndices{MOI.VectorOfVariables,S}())
        # display(ci_src)
        f_src = MOI.get(src, MOI.ConstraintFunction(), ci_src)
        if !allunique(f_src.variables)
            error("Cannot copy constraint `$(ci_src)` as variables constrained on creation because there are duplicate variables in the function `$(f_src)`",
                  ". Use `MOI.instantiate(SDPA.Optimizer, with_bridge_type = Float64)` ",
                  "to bridge this by creating slack variables.")
        elseif any(vi -> haskey(index_map, vi), f_src.variables)
            error("Cannot copy constraint `$(ci_src)` as variables constrained on creation because some variables of the function `$(f_src)` are in another constraint as well.",
                  ". Use `MOI.instantiate(SDPA.Optimizer, with_bridge_type = Float64)` ",
                  "to bridge constraints having the same variables by creating slack variables.")
        else
            set = MOI.get(src, MOI.ConstraintSet(), ci_src)::S
            vis_dest, ci_dest = _add_constrained_variables(dest, set)
            index_map[ci_src] = ci_dest
            for (vi_src, vi_dest) in zip(f_src.variables, vis_dest)
                index_map[vi_src] = vi_dest
            end
        end
    end
end

# const AFF = MOI.ScalarAffineFunction{T}
# const EQ = MOI.EqualTo{T} where {T}
# const AFFEQ = MOI.ConstraintIndex{AFF,EQ}

MOI.supports_incremental_interface(::Optimizer) = false

function MOI.copy_to(dest::Optimizer{T}, src::MOI.ModelLike) where {T}
    println("copy_to called")
    MOI.empty!(dest)
    index_map = MOI.Utilities.IndexMap()

    constrain_variables_on_creation(
        dest,
        src,
        index_map,
        MOI.Nonnegatives,
    )
    constrain_variables_on_creation(
        dest,
        src,
        index_map,
        MOI.PositiveSemidefiniteConeTriangle,
    )

    vis_src = MOI.get(src, MOI.ListOfVariableIndices())
    if length(vis_src) != length(index_map.var_map)
        error("Free variables are not supported",
              "",
              "")
    end
    cis_src = MOI.get(src, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{T},MOI.EqualTo{T}}())
    println("length(cis_src): $(length(cis_src))")
    funcs = Vector{MOI.ScalarAffineFunction{T}}(undef, length(cis_src))

    f(n) = (n>0) ? n*(n+1)÷2 : abs(n)
    N = sum([f(k) for k in dest.inner.data.block_dims])

    for (k, ci_src) in enumerate(cis_src)
        funcs[k] = MOI.get(src, MOI.CanonicalConstraintFunction(), ci_src)
        set = MOI.get(src, MOI.ConstraintSet(), ci_src)
        if !iszero(MOI.constant(funcs[k]))
            throw(MOI.ScalarFunctionConstantNotZero{
                T, MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}}(
                    MOI.constant(funcs[k])))
        end

        II = Int[]
        VV = T[]

        for term in funcs[k].terms

            if !iszero(term.coefficient)
                blk, i, j = varmap(dest, index_map[term.variable])
                blk_offset = sum(f.(dest.inner.data.block_dims[1:blk-1]))
                if dest.inner.data.block_dims[blk] > 0
                    ii = blk_offset + sum(1:i-1) + j
                else
                    ii = blk_offset + i
                end

                coef = term.coefficient

                if i != j
                    coef /= T(2)
                    coef *= T(sqrt(2.0))
                end
                push!(II, ii)
                push!(VV, coef)
            end
        end

        if !isempty(II)
            add_constraint!(dest.inner.data, sparsevec(II,VV,N), MOI.constant(set))
        end

        index_map[ci_src] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},MOI.EqualTo{T}}(k)
    end

    MOI.Utilities.pass_attributes(dest, src, index_map, vis_src)
    # Throw error for constraint attributes
    MOI.Utilities.pass_attributes(dest, src, index_map, cis_src)

    model_attributes = MOI.get(src, MOI.ListOfModelAttributesSet())

    if MOI.ObjectiveSense() in model_attributes
        sense = MOI.get(src, MOI.ObjectiveSense())
        objective_sign = sense == MOI.MIN_SENSE ? -1 : 1
        dest.objective_sense = objective_sign
        dest.inner.optimization_sense = -T(objective_sign)
    end

    if MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}() in model_attributes
        func = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}())
        obj = MOI.Utilities.canonical(func)

        dest.inner.obj_constant = obj.constant

        II = Int[]
        VV = T[]
        
        for term in obj.terms
            if !iszero(term.coefficient)
                blk, i, j = varmap(dest, index_map[term.variable])
                blk_offset = sum(f.(dest.inner.data.block_dims[1:blk-1]))
                coef = term.coefficient
                if dest.inner.data.block_dims[blk] > 0
                    ii = blk_offset + sum(1:i-1) + j
                else
                    ii = blk_offset + i
                end

                if i != j
                    coef /= T(2.0)
                    coef *= T(sqrt(2.0))
                end
                push!(II, ii)
                push!(VV, -T(objective_sign)*coef)
            end
        end

        set_objective!(dest.inner.data, sparsevec(II,VV,N))
    end

    return index_map
end

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    if opt.inner.status == :SolverNotCalled
        return MOI.OPTIMIZE_NOT_CALLED
    end
    if opt.inner.status == :Empty
        return MOI.OPTIMIZE_NOT_CALLED
    end

    if opt.inner.status == :Maximum_number_of_iterations_reached
        return MOI.ITERATION_LIMIT
    elseif opt.inner.status == :Solved
        return MOI.OPTIMAL
    elseif opt.inner.status == :Dual_infeasible
        return MOI.DUAL_INFEASIBLE
    elseif opt.inner.status == :Primal_infeasible
        return MOI.INFEASIBLE
    end
end

MOI.get(opt::Optimizer, ::MOI.ResultCount) = 1

function MOI.get(opt::Optimizer, ::MOI.ObjectiveValue)
    # MOI.check_result_index_bounds(opt, attr)
    return opt.inner.optimization_sense*dot(opt.inner.data.C, opt.inner.X) + opt.inner.obj_constant
end

function MOI.get(opt::Optimizer, ::MOI.DualObjectiveValue)
    # MOI.check_result_index_bounds(opt, attr)
    return opt.inner.optimization_sense*dot(opt.inner.data.b, opt.inner.y) + opt.inner.obj_constant
end

function MOI.get(::Optimizer, ::MOI.PrimalStatus)
    return MOI.FEASIBLE_POINT
end

function MOI.get(::Optimizer, ::MOI.DualStatus)
    return MOI.FEASIBLE_POINT
end

function MOI.get(opt::Optimizer{T}, attr::MOI.VariablePrimal, vi::MOI.VariableIndex) where {T}
    MOI.check_result_index_bounds(opt, attr)
    blk, i, j = varmap(opt, vi)

    f(n) = (n>0) ? n*(n+1)÷2 : abs(n)
    N = sum([f(k) for k in opt.inner.data.block_dims])

    blk_offset = sum(f.(opt.inner.data.block_dims[1:blk-1]))
    if opt.inner.data.block_dims[blk] > 0
        ii = blk_offset + sum(1:i-1) + j
    else
        ii = blk_offset + i
    end

    α = one(T)
    if i != j
        α = T(sqrt(2.0))
    end

    return opt.inner.X[ii]/α
end

function MOI.get(opt::Optimizer{T}, attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VectorOfVariables, S}) where {T, S<:SupportedSets}
    MOI.check_result_index_bounds(opt, attr)
    blk = opt.varmap[ci.value][1]
    val = opt.inner.X[opt.inner.data.block_indices[blk]]

    if opt.inner.data.block_dims[blk] > 0
        k = 1
        for i = 1:abs(opt.inner.data.block_dims[blk])
            for j = 1:i
                if i != j
                    val[k] *= T(1.0/sqrt(2.0))
                end
                k += 1
            end
        end
    end

    return val
end

function MOI.get(opt::Optimizer{T}, attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VectorOfVariables, S}) where {T, S<:SupportedSets}
    MOI.check_result_index_bounds(opt, attr)
    blk = opt.varmap[ci.value][1]
    val = opt.inner.S[opt.inner.data.block_indices[blk]]

    if opt.inner.data.block_dims[blk] > 0
        k = 1
        for i = 1:opt.inner.data.block_dims[blk]
            for j = 1:i
                if i != j
                    val[k] *= T(1.0/sqrt(2.0))
                end
                k += 1
            end
        end
    end

    return val
end

function MOI.get(opt::Optimizer, attr::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},MOI.EqualTo{T}}) where {T}
    MOI.check_result_index_bounds(opt, attr)
    if isempty(MOI.get(opt, MOI.CanonicalConstraintFunction(), ci).terms)
        return 0.0
    end
    return opt.inner.data.b[ci.value]
end

function MOI.get(opt::Optimizer{T}, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},MOI.EqualTo{T}}) where {T}
    MOI.check_result_index_bounds(opt, attr)
    return opt.inner.y[ci.value]
end
