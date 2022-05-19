
function scale!(α::T, A::Vector{T}) where {T<:Union{Float32,Float64}}
    println("scale called")
    BLAS.scal!(length(A), α, A, 1)
end

function scale!(α::BigFloat, A::Vector{BigFloat})
    @views A *= α
end

function scale!(α, A)
    @views A *= α
end