function get_l2_loss_function(loss_function, dataset)
    loss(θ) = mean(abs2, loss_function(dataset, θ))
    return loss
end

@inline null_additional_loss(phi, θ) = 0

function get_numeric_integral(pinnrep::NamedTuple)
    (; strategy, indvars, depvars, derivative, depvars, indvars, dict_indvars, dict_depvars) = pinnrep

    integral = (u, cord, phi, integrating_var_id, integrand_func, lb, ub, θ; strategy=strategy, indvars=indvars, depvars=depvars, dict_indvars=dict_indvars, dict_depvars=dict_depvars) -> begin
        function integration_(cord, lb, ub, θ)
            cord_ = cord
            function integrand_(x, p)
                ChainRulesCore.@ignore_derivatives @views(cord_[integrating_var_id]) .= x
                return integrand_func(cord_, p, phi, derivative, nothing, u, nothing)
            end
            prob_ = IntegralProblem(integrand_, lb, ub, θ)
            sol = solve(prob_, CubatureJLh(); reltol=1e-3, abstol=1e-3)[1]

            return sol
        end

        lb_ = zeros(size(lb)[1], size(cord)[2])
        ub_ = zeros(size(ub)[1], size(cord)[2])
        for (i, l) in enumerate(lb)
            if l isa Number
                ChainRulesCore.@ignore_derivatives lb_[i, :] = fill(l, 1, size(cord)[2])
            else
                ChainRulesCore.@ignore_derivatives lb_[i, :] = l(cord, θ, phi, derivative,
                                                                 nothing, u, nothing)
            end
        end
        for (i, u_) in enumerate(ub)
            if u_ isa Number
                ChainRulesCore.@ignore_derivatives ub_[i, :] = fill(u_, 1, size(cord)[2])
            else
                ChainRulesCore.@ignore_derivatives ub_[i, :] = u_(cord, θ, phi, derivative,
                                                                  nothing, u, nothing)
            end
        end
        integration_arr = Matrix{Float64}(undef, 1, 0)
        for i in 1:size(cord)[2]
            # ub__ = @Zygote.ignore getindex(ub_, :,  i)
            # lb__ = @Zygote.ignore getindex(lb_, :,  i)
            integration_arr = hcat(integration_arr,
                                   integration_(cord[:, i], lb_[:, i], ub_[:, i], θ))
        end
        return integration_arr
    end
end

"""
This function is only used for the first order derivative.
"""
forwarddiff(phi, t, εs, order, θ) = ForwardDiff.gradient(sum ∘ Base.Fix2(phi, θ), t)

function finitediff(phi, x, εs, order, θ)
    ε = εs[order]
    _epsilon = inv(first(ε[ε .!= zero(ε)]))
    ε = ChainRulesCore.@ignore_derivatives adapt(parameterless_type(x), ε)

    if any(x -> x != εs[1], εs)
        return (finitediff(phi, x .+ ε, @view(εs[1:(end - 1)]), order - 1, θ) .-
                finitediff(phi, x .- ε, @view(εs[1:(end - 1)]), order - 1, θ)) .*
               _epsilon ./ 2
    else
        finitediff(phi, x, ε, Val(order), θ, _epsilon)
    end
end

@inline function finitediff(phi, x, ε::AbstractVector{T}, ::Val{1}, θ,
                            h::T) where {T <: AbstractFloat}
    return (phi(x .+ ε, θ) .- phi(x .- ε, θ)) .* h ./ 2
end

@inline function finitediff(phi, x, ε::AbstractVector{T}, ::Val{2}, θ,
                            h::T) where {T <: AbstractFloat}
    return (phi(x .+ ε, θ) .+ phi(x .- ε, θ) .- 2 .* phi(x, θ)) .* h^2
end

@inline function finitediff(phi, x, ε::AbstractVector{T}, ::Val{3}, θ,
                            h::T) where {T <: AbstractFloat}
    return (phi(x .+ 2 .* ε, θ) .- 2 .* phi(x .+ ε, θ) .+ 2 .* phi(x .- ε, θ) -
            phi(x .- 2 .* ε, θ)) .* h^3 ./ 2
end

@inline function finitediff(phi, x, ε::AbstractVector{T}, ::Val{4}, θ,
                            h::T) where {T <: AbstractFloat}
    return (phi(x .+ 2 .* ε, θ) .- 4 .* phi(x .+ ε, θ) .+ 6 .* phi(x, θ) .-
            4 .* phi(x .- ε, θ) .+ phi(x .- 2 .* ε, θ)) .* h^4
end

function finitediff(phi, x, θ, dim::Int, order::Int)
    ε = ChainRulesCore.@ignore_derivatives get_ε(size(x, 1), dim, eltype(θ), order)
    _type = parameterless_type(ComponentArrays.getdata(θ))
    _epsilon = inv(first(ε[ε .!= zero(ε)]))

    ε = adapt(_type, ε)

    if order == 4
        return (phi(x .+ 2 .* ε, θ) .- 4 .* phi(x .+ ε, θ) .+ 6 .* phi(x, θ) .-
                4 .* phi(x .- ε, θ) .+ phi(x .- 2 .* ε, θ)) .* _epsilon^4
    elseif order == 3
        return (phi(x .+ 2 .* ε, θ) .- 2 .* phi(x .+ ε, θ, phi) .+ 2 .* phi(x .- ε, θ) -
                phi(x .- 2 .* ε, θ)) .* _epsilon^3 ./ 2
    elseif order == 2
        return (phi(x .+ ε, θ) .+ phi(x .- ε, θ) .- 2 .* phi(x, θ)) .* _epsilon^2
    elseif order == 1
        return (phi(x .+ ε, θ) .- phi(x .- ε, θ)) .* _epsilon ./ 2
    else
        error("The order $order is not supported!")
    end
end

function Base.getproperty(d::Symbolics.VarDomainPairing, var::Symbol)
    if var == :variables
        return getfield(d, :variables)
    elseif var == :domain
        return getfield(d, :domain)
    else
        idx = findfirst(v -> v.name === var, d.variables)
        domain = getfield(d, :domain)
        return Interval(infimum(domain)[idx], supremum(domain)[idx])
    end
end
