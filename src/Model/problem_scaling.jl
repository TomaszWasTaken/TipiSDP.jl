mutable struct ProblemScaling{T}
    normsA::Vector{T}
    normC::T
    normb::T

    function ProblemScaling{T}() where {T}
        return new(T[],
                   zero(T),
                   zero(T))
    end
end

# import Base: empty!
# import Base.empty!

function _empty!(scal::ProblemScaling{T}) where {T}
    # empty!(scal.normsA)
    scal.normsA = T[]
    scal.normC = NaN
    scal.normb = NaN
    return nothing
end

function add_lp_block!(scal::ProblemScaling{T}) where {T}
    push!(scal.normsA, one(T))
    return nothing
end

function add_sdp_block!(scal::ProblemScaling{T}) where {T}
    push!(scal.normsA, one(T))
    return nothing
end
