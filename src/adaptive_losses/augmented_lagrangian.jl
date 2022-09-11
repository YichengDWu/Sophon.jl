mutable struct AugmentedLagrangian{T} <: NeuralPDE.AbstractAdaptiveLoss
    bc_loss_weights::Vector{T}
    additional_loss_weights::Vector{T}
    ϵ::T
    μ_max::T
    η::T
    μ::Float64
end

function AugmentedLagrangian(pde_loss_lenghth::Int, bc_loss_length::Int,
    additional_loss_length::Int=0;
    ϵ::Float64 = 1e-8,
    μ_max::Float64=1e4)

    AugmentedLagrangian{Float64}(zeros(Float64, bc_loss_length),
                                 zeros(Float64, additional_loss_length),
                                 ϵ,
                                 η,
                                 μ_max,
                                 0.0,
                                 1.0)
end

function generate_full_loss_function(pinnrep::NeuralPDE.PINNRepresentation,
                                     adaloss::AugmentedLagrangian,
                                     pde_loss_functions, bc_loss_functions,
                                     additional_loss_function::Nothing)

    function full_loss_function(θ, _)
        pde_loss = sum(pde_loss_function(θ) for pde_loss_function in pde_loss_functions)
        bc_losses = [bc_loss_function(θ) for bc_loss_function in bc_loss_functions]

        bc_loss = sum(adaloss.bc_loss_weights .* bc_losses)
        penalty = sum(abs2, bc_losses)

        loss = pde_loss + bc_loss + adaloss.μ/2 * penalty

        ChainRulesCore.@ignore_derivatives begin
            if √penalty ≥ 0.25*adaloss.η && √penalty > adaloss.ϵ
                adaloss.μ = min(adaloss.μ*2.0, adaloss.μ_max)
                adaloss.bc_loss_weights .= adaloss.bc_loss_weights .+ adaloss.μ .* bc_losses
            end
            adaloss.η  = √penalty
        end
        return loss
    end
end
