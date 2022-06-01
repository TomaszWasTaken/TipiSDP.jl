using TipiSDP, Test, LinearAlgebra, SparseArrays

@testset "ProblemData" begin
    prb_data = TipiSDP.ProblemData{Float64}()
    @test TipiSDP.is_empty(prb_data) == true
    TipiSDP.add_lp_block!(prb_data, 3)
    @test prb_data.block_dims == [-3] && 
          size(prb_data.A) == (3, 0) &&
          size(prb_data.C) == (3,)
    TipiSDP.add_sdp_block!(prb_data, 2)
    @test prb_data.block_dims == [-3, 2] &&
          size(prb_data.A) == (3+(2*(2+1)÷2), 0) &&
          size(prb_data.C) == (3+(2*(2+1)÷2),)
    TipiSDP.set_objective!(prb_data, [1.0,2.0,3.0,4.0,5.0,6.0])
    @test prb_data.C == [1.0,2.0,3.0,4.0,5.0,6.0]
    TipiSDP.add_constraint!(prb_data, [1.0,2.0,3.0,4.0,5.0,6.0], 1.0)
    @test prb_data.A == sparse([1.0 2.0 3.0 4.0 5.0 6.0]') &&
          prb_data.b == [1.0] &&
          prb_data.m == 1
    TipiSDP._empty!(prb_data)
    @test TipiSDP.is_empty(prb_data) == true
end