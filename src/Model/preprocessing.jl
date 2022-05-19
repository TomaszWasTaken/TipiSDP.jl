function compute_spattern(prb::InnerModel{T}) where {T}

    Csp = sparse(prb.data.C)
    s = Set{Tuple{Int,T}}()

    n, m = size(prb.data.A)

    for (ind,elem) in enumerate(Csp.nzind)
        push!(s, (elem,one(T)))
    end

    for j = 1:m
        for (ind,elem) in enumerate(prb.data.A[:,j].nzind)
            if !(elem in s)
                push!(s, (elem, prb.data.A[ind,j]))
            end
        end
    end

    II = Int[]
    JJ = Int[]
    VV = T[]

    N = (isqrt(8*n+1)-1)÷2

    for (k, val) in s
        ii, jj = ind2cart(k)
        if ii == jj
            push!(II, ii)
            push!(JJ, jj)
            push!(VV, 10.0*one(T))
        else
            push!(II, ii)
            push!(JJ, jj)
            push!(VV, one(T))
            push!(II, jj)
            push!(JJ, ii)
            push!(VV, one(T))
        end
    end

    Ssp = sparse(II, JJ, VV, N, N)
end