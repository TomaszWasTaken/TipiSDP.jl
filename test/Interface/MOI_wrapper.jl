module TestTipiSDP

using Test
using MathOptInterface
import TipiSDP

const MOI = MathOptInterface

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_solver_name()
    @test MOI.get(TipiSDP.Optimizer(), MOI.SolverName()) == "TipiSDP"
end

function test_options()
    param = MOI.RawOptimizerAttribute("bad_option")
    err = MOI.UnsupportedAttribute(param)
    @test_throws err MOI.set(
        TipiSDP.Optimizer(),
        MOI.RawOptimizerAttribute("bad_option"),
        0,
    )
end

function test_runtests()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.instantiate(TipiSDP.Optimizer, with_bridge_type = Float64),
    )
    # `Variable.ZerosBridge` makes dual needed by some tests fail.
    MOI.Bridges.remove_bridge(
        model.optimizer,
        MathOptInterface.Bridges.Variable.ZerosBridge{Float64},
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            rtol = 1e-3,
            atol = 1e-3,
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.VariableBasisStatus,
                MOI.ObjectiveBound,
                MOI.SolverVersion,
            ],
        ),
        include = String[], #"test_conic_SecondOrderCone_negative_post_bound_2"
        exclude = String[
            # Unable to bridge RotatedSecondOrderCone to PSD because the dimension is too small: got 2, expected >= 3.
            "test_conic_SecondOrderCone_INFEASIBLE",
            "test_constraint_PrimalStart_DualStart_SecondOrderCone",
            # Expression: MOI.get(model, MOI.TerminationStatus()) == MOI.INFEASIBLE
            #  Evaluated: MathOptInterface.INFEASIBLE_OR_UNBOUNDED == MathOptInterface.INFEASIBLE
            "test_conic_NormInfinityCone_INFEASIBLE",
            "test_conic_NormOneCone_INFEASIBLE",
            # Incorrect objective
            # See https://github.com/jump-dev/MathOptInterface.jl/issues/1759
            "test_unbounded_MIN_SENSE",
            "test_unbounded_MIN_SENSE_offset",
            "test_unbounded_MAX_SENSE",
            "test_unbounded_MAX_SENSE_offset",
            "test_infeasible_MAX_SENSE",
            "test_infeasible_MAX_SENSE_offset",
            "test_infeasible_MIN_SENSE",
            "test_infeasible_MIN_SENSE_offset",
            "test_infeasible_affine_MAX_SENSE",
            "test_infeasible_affine_MAX_SENSE_offset",
            "test_infeasible_affine_MIN_SENSE",
            "test_infeasible_affine_MIN_SENSE_offset",
            # TODO remove when PR merged
            # See https://github.com/jump-dev/MathOptInterface.jl/pull/1769
            "test_objective_ObjectiveFunction_blank",
            # FIXME investigate
            #  Expression: isapprox(MOI.get(model, MOI.ObjectiveValue()), T(2), config)
            #   Evaluated: isapprox(5.999999984012059, 2.0, ...
            "test_modification_delete_variables_in_a_batch",
            # FIXME investigate
            #  Expression: isapprox(MOI.get(model, MOI.ObjectiveValue()), objective_value, config)
            #   Evaluated: isapprox(-2.1881334077988868e-7, 5.0, ...
            "test_objective_qp_ObjectiveFunction_edge_case",
            # FIXME investigate
            #  Expression: isapprox(MOI.get(model, MOI.ObjectiveValue()), objective_value, config)
            #   Evaluated: isapprox(-2.1881334077988868e-7, 5.0, ...
            "test_objective_qp_ObjectiveFunction_zero_ofdiag",
            # FIXME investigate
            #  Expression: isapprox(MOI.get(model, MOI.ConstraintPrimal(), index), solution_value, config)
            #   Evaluated: isapprox(2.5058846553349667e-8, 1.0, ...
            "test_variable_solve_with_lowerbound",
            # NOT DEFINED ?
            "test_conic_GeometricMeanCone_VectorAffineFunction_2",
            "test_conic_GeometricMeanCone_VectorOfVariables_2",
            "test_solve_result_index",
            "test_objective_FEASIBILITY_SENSE_clears_objective",
            "test_objective_ObjectiveFunction_VariableIndex",
            "test_objective_ObjectiveFunction_constant",
            "test_objective_ObjectiveFunction_duplicate_terms",
            "test_modification_set_singlevariable_lessthan",
            "test_modification_transform_singlevariable_lessthan",
            "test_modification_coef_scalar_objective",
            "test_model_copy_to_UnsupportedAttribute",
            "test_modification_const_vectoraffine_zeros",
            "test_modification_delete_variable_with_single_variable_obj",
            "test_modification_const_scalar_objective"
            # "test_variable_VariableName",
            # "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            # "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_VariableIndex_LessThan_max",
            # "test_quadratic_nonhomogeneous",
            # "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_EqualTo_lower",
            # "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_EqualTo_upper",
            # "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_GreaterThan",
            # "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_Interval_lower",
            # "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_Interval_upper",
            # "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_LessThan",
            # "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_VariableIndex_LessThan",
            # "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_VariableIndex_LessThan_max",
            # "test_solve_result_index",
            # "test_objective_FEASIBILITY_SENSE_clears_objective"
        ],
    )
    return
end

end  # module

TestTipiSDP.runtests()