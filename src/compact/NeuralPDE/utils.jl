function get_bounds(domains::Vector{Symbolics.VarDomainPairing}, eqs, bcs, eltypeθ,
                    dict_indvars, dict_depvars)
    dict_span = Dict([Symbol(d.variables) => [infimum(d.domain), supremum(d.domain)]
                      for d in domains])
    pde_args = NeuralPDE.get_argument(eqs, dict_indvars, dict_depvars)
    pde_bounds = map(pde_args) do pde_arg
        bds = mapreduce(s -> get(dict_span, s, fill(s, 2)), hcat, pde_arg)
        bds = eltypeθ.(bds)
        return bds[1, :], bds[2, :]
    end

    bound_args = NeuralPDE.get_argument(bcs, dict_indvars, dict_depvars)
    bcs_bounds = map(bound_args) do bound_arg
        bds = mapreduce(s -> get(dict_span, s, fill(s, 2)), hcat, bound_arg)
        bds = eltypeθ.(bds)
        return bds[1, :], bds[2, :]
    end
    return pde_bounds, bcs_bounds
end

function get_bounds(d::Domain)
    return infimum(d), supremum(d)
end

function get_bounds(d::Interval)
    return [infimum(d)], [supremum(d)]
end

function get_bounds(pde::Sophon.PDESystem)
    pde_bounds = map(pde.eqs) do eq
        return get_bounds(eq[2])
    end
    bcs_bounds = map(pde.bcs) do bc
        return get_bounds(bc[2])
    end
    return pde_bounds, bcs_bounds
end

function get_bounds(pde::NeuralPDE.PDESystem)
    (; eqs, bcs, domain, ivs, dvs) = pde
    _, _, dict_indvars, dict_depvars, _ = NeuralPDE.get_vars(ivs, dvs)
    bounds = get_bounds(domain, eqs, bcs, Float64, dict_indvars, dict_depvars)
    return bounds
end

function get_l2_loss_function(loss_function, dataset)
    loss(θ) = mean(abs2, loss_function(dataset, θ))
    return loss
end

@inline null_additional_loss(phi, θ) = 0
"""
```julia
:((cord, θ) -> begin
  #= ... =#
  #= ... =#
  begin
      (θ_depvar_1, θ_depvar_2) = (θ.depvar_1, θ.depvar_2)
      (phi_depvar_1, phi_depvar_2) = (phi.depvar_1, phi.depvar_2)
      let (x, y) = (cord[1], cord[2])
          [
              (+)(derivative(phi_depvar_1, u, [x, y], [[ε, 0.0]], 1, θ_depvar_1),
                  (*)(4, derivative(phi_depvar_1, u, [x, y], [[0.0, ε]], 1, θ_depvar_2))) -
              0,
              (+)(derivative(phi_depvar_2, u, [x, y], [[ε, 0.0]], 1, θ_depvar_2),
                  (*)(9, derivative(phi_depvar_2, u, [x, y], [[0.0, ε]], 1, θ_depvar_1))) -
              0,
          ]
      end
  end end)
```
"""
function build_symbolic_loss_function(pinnrep::NamedTuple{names},
                                      eq::Symbolics.Equation) where {names}
    (; depvars, dict_depvars, dict_depvar_input, derivative, multioutput, dict_indvars) = pinnrep

    loss_function, pos, values = parse_equation(pinnrep, eq)
    this_eq_pair = pair(eq, depvars, dict_depvar_input)
    this_eq_indvars = unique(vcat([getindex(this_eq_pair, v) for v in keys(this_eq_pair)]...))

    vars = :(cord, θ, pfs)
    ex = Expr(:block)
    if :pvs ∈ names
        (; pinn, cord_branch_net) = pinnrep

        push!(ex.args, Expr(:(=), :deeponet, pinn.phi))
        push!(ex.args, Expr(:(=), :derivative, derivative))
        push!(ex.args, Expr(:(=), :cord_branch_net, cord_branch_net))
        push!(ex.args,
              Expr(:(=), :(get_pfs_output(x::AbstractMatrix)),
                   :(ChainRulesCore.ignore_derivatives(mapreduce(f -> f.(x), vcat, pfs)))))
        push!(ex.args,
              Expr(:(=), :(get_pfs_output(x::AbstractVector...)),
                   :(ChainRulesCore.ignore_derivatives(mapreduce(f -> reshape(f.(x...), 1,
                                                                              :), vcat,
                                                                 pfs)))))
        push!(ex.args,
              Expr(:(=), :(branch_net_input),
                   :(transpose(get_pfs_output(cord_branch_net...)))))
        push!(ex.args, Expr(:(=), :(phi(x_, θ_)), :(deeponet((branch_net_input, x_), θ_))))

    else
        phi = pinnrep.phi
        push!(ex.args, Expr(:(=), :phi, phi))
        push!(ex.args, Expr(:(=), :derivative, derivative))
    end

    if multioutput
        θs = Symbol[]
        phis = Symbol[]
        for v in depvars
            push!(θs, :($(Symbol(:θ, :_, v))))
            push!(phis, :($(Symbol(:phi, :_, v))))
        end

        expr_θ = Expr[]
        expr_phi = Expr[]

        for u in depvars
            push!(expr_θ, :(θ.$(u)))
            push!(expr_phi, :(phi.$(u)))
        end

        vars_θ = Expr(:(=), NeuralPDE.build_expr(:tuple, θs),
                      NeuralPDE.build_expr(:tuple, expr_θ))
        push!(ex.args, vars_θ)

        vars_phi = Expr(:(=), NeuralPDE.build_expr(:tuple, phis),
                        NeuralPDE.build_expr(:tuple, expr_phi))
        push!(ex.args, vars_phi)
    end

    eq_pair_expr = Expr[]
    if pos isa Int
        v = first(keys(this_eq_pair))
        ivars_l = convert(Vector{Any}, deepcopy(this_eq_pair[v]))
        ivars_r = convert(Vector{Any}, deepcopy(this_eq_pair[v]))
        ivars_l[pos] = :(zero(cord[[1], :]) .+ $(values[1]))
        ivars_r[pos] = :(zero(cord[[1], :]) .+ $(values[2]))
        push!(eq_pair_expr, :($(Symbol(:cord, :_, :($v), :_l)) = vcat($(ivars_l...))))
        push!(eq_pair_expr, :($(Symbol(:cord, :_, :($v), :_r)) = vcat($(ivars_r...))))
        this_eq_indvars = this_eq_indvars[setdiff(1:length(this_eq_indvars), pos)]
    else
        for v in keys(this_eq_pair)
            push!(eq_pair_expr,
                  :($(Symbol(:cord, :_, :($v))) = vcat($(this_eq_pair[v]...))))
        end
    end
    vcat_expr = Expr(:block, :($(eq_pair_expr...)))
    vcat_expr_loss_functions = Expr(:block, vcat_expr, loss_function)

    indvars_ex = [:($:cord[[$i], :]) for (i, x) in enumerate(this_eq_indvars)]
    left_arg_pairs, right_arg_pairs = this_eq_indvars, indvars_ex
    vars_eq = Expr(:(=), NeuralPDE.build_expr(:tuple, left_arg_pairs),
                   NeuralPDE.build_expr(:tuple, right_arg_pairs))

    let_ex = Expr(:let, vars_eq, vcat_expr_loss_functions)
    push!(ex.args, let_ex)
    return vars, ex
end

function build_loss_function(pinnrep::NamedTuple, eq::Symbolics.Equation, i)
    vars, ex = build_symbolic_loss_function(pinnrep, eq)
    expr = Expr(:function,
                Expr(:call, Symbol(:residual_function_, i), vars.args[1], vars.args[2],
                     :($(Expr(:kw, vars.args[3], :nothing)))), ex)
    return eval(expr)
end

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

function parse_equation(pinnrep::NamedTuple, eq)
    eq_lhs = isequal(NeuralPDE.expand_derivatives(eq.lhs), 0) ? eq.lhs :
             NeuralPDE.expand_derivatives(eq.lhs)
    eq_rhs = isequal(NeuralPDE.expand_derivatives(eq.rhs), 0) ? eq.rhs :
             NeuralPDE.expand_derivatives(eq.rhs)
    left_expr = NeuralPDE.toexpr(eq_lhs)
    right_expr = NeuralPDE.toexpr(eq_rhs)

    tran_left_expr = transform_expression(pinnrep, NeuralPDE.toexpr(eq_lhs))
    tran_right_expr = transform_expression(pinnrep, NeuralPDE.toexpr(eq_rhs))
    dot_left_expr = NeuralPDE._dot_(tran_left_expr)
    dot_right_expr = NeuralPDE._dot_(tran_right_expr)

    if is_periodic_bc(pinnrep.bcs, eq, pinnrep.depvars, left_expr, right_expr)
        pos = findfirst((left_expr.args[2:end] .!== right_expr.args[2:end])) # Assume the pericity is defined on n-1 hyperplanes with one depvar
        values = (left_expr.args[pos + 1], right_expr.args[pos + 1])

        dot_left_expr, dot_right_expr = parse_periodic_condition(dot_left_expr,
                                                                 dot_right_expr)
        return loss_function = :($dot_left_expr .- $dot_right_expr), pos, values
    else
        return loss_function = :($dot_left_expr .- $dot_right_expr), nothing, nothing
    end
end

function is_periodic_bc(bcs::Vector{<:Symbolics.Equation}, eq, depvars, left_expr::Expr,
                        right_expr::Expr)
    eq ∉ bcs && return false
    return left_expr.args[1] ∈ depvars && left_expr.args[1] === right_expr.args[1]
end

function is_periodic_bc(bcs::Vector{<:Pair{<:Symbolics.Equation, <:DomainSets.Domain}}, eq,
                        depvars, left_expr::Expr, right_expr::Expr)
    eq ∉ bcs[1] && return false
    return left_expr.args[1] ∈ depvars && left_expr.args[1] === right_expr.args[1]
end

function is_periodic_bc(bcs, eq, depvars, left_expr::Number, right_expr::Expr)
    return false
end

function is_periodic_bc(bcs, eq, depvars, left_expr::Expr, right_expr::Number)
    return false
end

function parse_periodic_condition(left_expr, right_expr)
    left_expr.args[2] = Symbol(left_expr.args[2], :_l)
    right_expr.args[2] = Symbol(right_expr.args[2], :_r)
    return left_expr, right_expr
end

function transform_expression(pinnrep::NamedTuple, ex)
    if ex isa Expr
        ex = _transform_expression(pinnrep, ex)
    end
    return ex
end

function _transform_expression(pinnrep::NamedTuple{names}, ex::Expr) where {names}
    (; indvars, depvars, dict_indvars, dict_depvars, dict_depvar_input, multioutput, fdtype) = pinnrep
    fdtype = fdtype
    dict_pmdepvars = :dict_pmdepvars in names ? pinnrep.dict_pmdepvars :
                     Dict{Symbol, Symbol}()
    _args = ex.args
    for (i, e) in enumerate(_args)
        if !(e isa Expr)
            if e in keys(dict_depvars)
                ex.args = if !multioutput
                    [:($(Expr(:$, :phi))), Symbol(:cord, :_, e), :θ]
                else
                    [
                        :($(Expr(:$, Symbol(:phi, :_, e)))),
                        Symbol(:cord, :_, e),
                        Symbol(:θ, :_, e),
                    ]
                end
                break
            elseif e in keys(dict_pmdepvars)
                ex.args[1] = :($(Expr(:$, :get_pfs_output)))
                break
            elseif e isa NeuralPDE.Differential
                derivative_variables = Symbol[]
                order = 0
                while (_args[1] isa NeuralPDE.Differential)
                    order += 1
                    push!(derivative_variables, NeuralPDE.toexpr(_args[1].x))
                    _args = _args[2].args
                end
                depvar = _args[1]
                num_depvar = dict_depvars[depvar]
                indvars = _args[2:end]
                dict_interior_indvars = Dict([indvar .=> j
                                              for (j, indvar) in enumerate(dict_depvar_input[depvar])])
                dim_l = length(dict_interior_indvars)

                εs = [get_ε(dim_l, d, fdtype, order) for d in 1:dim_l]
                undv = [dict_interior_indvars[d_p] for d_p in derivative_variables]
                εs_dnv = [εs[d] for d in undv]

                ex.args = if !multioutput
                    [
                        :($(Expr(:$, :derivative))),
                        :phi,
                        Symbol(:cord, :_, depvar),
                        εs_dnv,
                        order,
                        :θ,
                    ]
                else
                    [
                        :($(Expr(:$, :derivative))),
                        Symbol(:phi, :_, depvar),
                        Symbol(:cord, :_, depvar),
                        εs_dnv,
                        order,
                        Symbol(:θ, :_, depvar),
                    ]
                end
                break
            end
        else
            ex.args[i] = _transform_expression(pinnrep, ex.args[i])
        end
    end
    return ex
end

"""
Finds which dependent variables are being used in an equation.
"""
function pair(eq, depvars, dict_depvar_input)
    expr = NeuralPDE.NeuralPDE.toexpr(eq)
    pair_ = map(depvars) do depvar
        if !isempty(NeuralPDE.find_thing_in_expr(expr, depvar))
            depvar => dict_depvar_input[depvar]
        end
    end
    return Dict(filter(p -> p !== nothing, pair_))
end

function get_ε(dim, der_num, fdtype, order)
    epsilon = ^(eps(fdtype), one(fdtype) / (2 + order))
    ε = zeros(fdtype, dim)
    ε[der_num] = epsilon
    return ε
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
