mutable struct Settings{T}
    silent::Bool
    
    solution_relgap::T
    solution_φ::T

    primal_infeas::T
    dual_infeas::T

    maxIter::Int

    function Settings{T}() where {T}
        return new(false,
                   T(1e-7),
                   T(1e-7),
                   T(1e8),
                   T(1e8),
                   50)
    end
end

function set_solution_relgap(s::Settings{T}, val) where {T}
    s.solution_relgap = T(val)
end

function set_solution_φ(s::Settings{T}, val) where {T}
    s.solution_φ = T(val)
end

function set_maxIter(s::Settings, val)
    s.maxIter = val
end