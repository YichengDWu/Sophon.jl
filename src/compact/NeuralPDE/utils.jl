θ = gensym("θ")

function get_bounds(domains, eqs, bcs, eltypeθ, dict_indvars, dict_depvars)
    dict_span = Dict([Symbol(d.variables) => [infimum(d.domain), supremum(d.domain)]
                      for d in domains])
    pde_args = NeuralPDE.get_argument(eqs, dict_indvars, dict_depvars)

    pde_bounds = map(pde_args) do pd
        span = map(p -> get(dict_span, p, p), pd)
        return map(s -> adapt(eltypeθ, s), span)
    end

    bound_args = NeuralPDE.get_argument(bcs, dict_indvars, dict_depvars)
    dict_span = Dict([Symbol(d.variables) => [infimum(d.domain), supremum(d.domain)]
                      for d in domains])

    bcs_bounds = map(bound_args) do bt
        span = map(b -> get(dict_span, b, b), bt)
        return map(s -> adapt(eltypeθ, s), span)
    end
    return pde_bounds, bcs_bounds
end

function get_l2_loss_function(loss_function, dataset)
    loss(θ) = mean(abs2, loss_function(dataset, θ))
    return loss
end

@inline null_additional_loss(phi, θ) = 0

function build_symbolic_loss_function(pinnrep::NamedTuple, eqs;
                                      eq_params=SciMLBase.NullParameters(),
                                      param_estim=false, default_p=nothing,
                                      bc_indvars=pinnrep.indvars, integrand=nothing,
                                      dict_transformation_vars=nothing,
                                      transformation_vars=nothing,
                                      integrating_depvars=pinnrep.depvars)
    (; indvars, depvars, dict_indvars, dict_depvars, dict_depvar_input, phi, derivative, integral, multioutput, init_params, strategy, eq_params, param_estim, default_p) = pinnrep

    eltypeθ = eltype(pinnrep.init_params)

    if integrand isa Nothing
        loss_function = parse_equation(pinnrep, eqs)
        this_eq_pair = NeuralPDE.pair(eqs, depvars, dict_depvars, dict_depvar_input)
        this_eq_indvars = unique(vcat(values(this_eq_pair)...))
    else
        this_eq_pair = Dict(map(intvars -> dict_depvars[intvars] => dict_depvar_input[intvars],
                                integrating_depvars))
        this_eq_indvars = transformation_vars isa Nothing ?
                          unique(vcat(values(this_eq_pair)...)) : transformation_vars
        loss_function = integrand
    end

    vars = :(cord, $θ, phi, derivative, integral, u, p)
    ex = Expr(:block)
    if multioutput
        θ_nums = Symbol[]
        phi_nums = Symbol[]
        for v in depvars
            num = dict_depvars[v]
            push!(θ_nums, :($(Symbol(:($θ), num))))
            push!(phi_nums, :($(Symbol(:phi, num))))
        end

        expr_θ = Expr[]
        expr_phi = Expr[]

        acum = [0; accumulate(+, map(length, init_params))]
        sep = [(acum[i] + 1):acum[i + 1] for i in 1:(length(acum) - 1)]

        for u in depvars
            push!(expr_θ, :($θ.$(u)))
            push!(expr_phi, :(phi.$(u)))
        end

        vars_θ = Expr(:(=), NeuralPDE.build_expr(:tuple, θ_nums),
                      NeuralPDE.build_expr(:tuple, expr_θ))
        push!(ex.args, vars_θ)

        vars_phi = Expr(:(=), NeuralPDE.build_expr(:tuple, phi_nums),
                        NeuralPDE.build_expr(:tuple, expr_phi))
        push!(ex.args, vars_phi)
    end

    if eq_params != SciMLBase.NullParameters() && param_estim == false
        params_symbols = Symbol[]
        expr_params = Expr[]
        for (i, eq_param) in enumerate(eq_params)
            push!(expr_params, :(ArrayInterfaceCore.allowed_getindex(p, ($i):($i))))
            push!(params_symbols, Symbol(:($eq_param)))
        end
        params_eq = Expr(:(=), NeuralPDE.build_expr(:tuple, params_symbols),
                         NeuralPDE.build_expr(:tuple, expr_params))
        push!(ex.args, params_eq)
    end

    eq_pair_expr = Expr[]
    for i in keys(this_eq_pair)
        push!(eq_pair_expr, :($(Symbol(:cord, :($i))) = vcat($(this_eq_pair[i]...))))
    end
    vcat_expr = Expr(:block, :($(eq_pair_expr...)))
    vcat_expr_loss_functions = Expr(:block, vcat_expr, loss_function) # TODO rename

    if strategy isa QuadratureTraining
        indvars_ex = get_indvars_ex(bc_indvars)
        left_arg_pairs, right_arg_pairs = this_eq_indvars, indvars_ex
        vars_eq = Expr(:(=), NeuralPDE.build_expr(:tuple, left_arg_pairs),
                       NeuralPDE.build_expr(:tuple, right_arg_pairs))
    else
        indvars_ex = [:($:cord[[$i], :]) for (i, x) in enumerate(this_eq_indvars)]
        left_arg_pairs, right_arg_pairs = this_eq_indvars, indvars_ex
        vars_eq = Expr(:(=), NeuralPDE.build_expr(:tuple, left_arg_pairs),
                       NeuralPDE.build_expr(:tuple, right_arg_pairs))
    end

    if !(dict_transformation_vars isa Nothing)
        transformation_expr_ = Expr[]

        for (i, u) in dict_transformation_vars
            push!(transformation_expr_, :($i = $u))
        end
        transformation_expr = Expr(:block, :($(transformation_expr_...)))
        vcat_expr_loss_functions = Expr(:block, transformation_expr, vcat_expr,
                                        loss_function)
    end
    let_ex = Expr(:let, vars_eq, vcat_expr_loss_functions)
    push!(ex.args, let_ex)
    return expr_loss_function = :(($vars) -> begin $ex end)
end

function build_loss_function(pinnrep::NamedTuple, eqs, bc_indvars)
    (; eq_params, param_estim, default_p, phi, derivative, integral) = pinnrep

    bc_indvars = bc_indvars === nothing ? pinnrep.indvars : bc_indvars

    expr_loss_function = build_symbolic_loss_function(pinnrep, eqs; bc_indvars=bc_indvars,
                                                      eq_params=eq_params,
                                                      param_estim=param_estim,
                                                      default_p=default_p)
    u = NeuralPDE.get_u()
    _loss_function = NeuralPDE.NeuralPDE.@RuntimeGeneratedFunction(expr_loss_function)
    loss_function = (cord, θ) -> begin _loss_function(cord, θ, phi, derivative, integral, u,
                                                      default_p) end
    return loss_function
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
    eq_lhs = isequal(expand_derivatives(eq.lhs), 0) ? eq.lhs : expand_derivatives(eq.lhs)
    eq_rhs = isequal(expand_derivatives(eq.rhs), 0) ? eq.rhs : expand_derivatives(eq.rhs)
    left_expr = transform_expression(pinnrep, toexpr(eq_lhs))
    right_expr = transform_expression(pinnrep, toexpr(eq_rhs))
    left_expr = NeuralPDE._dot_(left_expr)
    right_expr = NeuralPDE._dot_(right_expr)
    return loss_func = :($left_expr .- $right_expr)
end

function transform_expression(pinnrep::NamedTuple, ex; is_integral=false,
                              dict_transformation_vars=nothing, transformation_vars=nothing)
    if ex isa Expr
        ex = _transform_expression(pinnrep, ex; is_integral=is_integral,
                                   dict_transformation_vars=dict_transformation_vars,
                                   transformation_vars=transformation_vars)
    end
    return ex
end

function _transform_expression(pinnrep::NamedTuple, ex; is_integral=false,
                               dict_transformation_vars=nothing,
                               transformation_vars=nothing)
    (; indvars, depvars, dict_indvars, dict_depvars, dict_depvar_input, multioutput, strategy, phi, derivative, integral, init_params) = pinnrep
    flat_init_params = init_params
    eltypeθ = eltype(flat_init_params)

    _args = ex.args
    for (i, e) in enumerate(_args)
        if !(e isa Expr)
            if e in keys(dict_depvars)
                depvar = _args[1]
                num_depvar = dict_depvars[depvar]
                indvars = _args[2:end]
                var_ = is_integral ? :(u) : :($(Expr(:$, :u)))
                ex.args = if !multioutput
                    [var_, Symbol(:cord, num_depvar), :($θ), :phi]
                else
                    [
                        var_,
                        Symbol(:cord, num_depvar),
                        Symbol(:($θ), num_depvar),
                        Symbol(:phi, num_depvar),
                    ]
                end
                break
            elseif e isa ModelingToolkit.Differential
                derivative_variables = Symbol[]
                order = 0
                while (_args[1] isa ModelingToolkit.Differential)
                    order += 1
                    push!(derivative_variables, toexpr(_args[1].x))
                    _args = _args[2].args
                end
                depvar = _args[1]
                num_depvar = dict_depvars[depvar]
                indvars = _args[2:end]
                dict_interior_indvars = Dict([indvar .=> j
                                              for (j, indvar) in enumerate(dict_depvar_input[depvar])])
                dim_l = length(dict_interior_indvars)

                var_ = is_integral ? :(derivative) : :($(Expr(:$, :derivative)))
                εs = [NeuralPDE.get_ε(dim_l, d, eltypeθ) for d in 1:dim_l]
                undv = [dict_interior_indvars[d_p] for d_p in derivative_variables]
                εs_dnv = [εs[d] for d in undv]

                ex.args = if !multioutput
                    [var_, :phi, :u, Symbol(:cord, num_depvar), εs_dnv, order, :($θ)]
                else
                    [
                        var_,
                        Symbol(:phi, num_depvar),
                        :u,
                        Symbol(:cord, num_depvar),
                        εs_dnv,
                        order,
                        Symbol(:($θ), num_depvar),
                    ]
                end
                break
            elseif e isa Symbolics.Integral
                if _args[1].domain.variables isa Tuple
                    integrating_variable_ = collect(_args[1].domain.variables)
                    integrating_variable = toexpr.(integrating_variable_)
                    integrating_var_id = [dict_indvars[i] for i in integrating_variable]
                else
                    integrating_variable = toexpr(_args[1].domain.variables)
                    integrating_var_id = [dict_indvars[integrating_variable]]
                end

                integrating_depvars = []
                integrand_expr = _args[2]
                for d in depvars
                    d_ex = find_thing_in_expr(integrand_expr, d)
                    if !isempty(d_ex)
                        push!(integrating_depvars, d_ex[1].args[1])
                    end
                end

                lb, ub = get_limits(_args[1].domain.domain)
                lb, ub, _args[2], dict_transformation_vars, transformation_vars = transform_inf_integral(lb,
                                                                                                         ub,
                                                                                                         _args[2],
                                                                                                         integrating_depvars,
                                                                                                         dict_depvar_input,
                                                                                                         dict_depvars,
                                                                                                         integrating_variable,
                                                                                                         eltypeθ)

                num_depvar = map(int_depvar -> dict_depvars[int_depvar],
                                 integrating_depvars)
                integrand_ = transform_expression(pinnrep, _args[2]; is_integral=false,
                                                  dict_transformation_vars=dict_transformation_vars,
                                                  transformation_vars=transformation_vars)
                integrand__ = NeuralPDE._dot_(integrand_)

                integrand = build_symbolic_loss_function(pinnrep, nothing;
                                                         integrand=integrand__,
                                                         integrating_depvars=integrating_depvars,
                                                         eq_params=SciMLBase.NullParameters(),
                                                         dict_transformation_vars=dict_transformation_vars,
                                                         transformation_vars=transformation_vars,
                                                         param_estim=false,
                                                         default_p=nothing)
                # integrand = repr(integrand)
                lb = toexpr.(lb)
                ub = toexpr.(ub)
                ub_ = []
                lb_ = []
                for l in lb
                    if l isa Number
                        push!(lb_, l)
                    else
                        l_expr = NeuralPDE.build_symbolic_loss_function(pinnrep, nothing;
                                                                        integrand=NeuralPDE._dot_(l),
                                                                        integrating_depvars=integrating_depvars,
                                                                        param_estim=false,
                                                                        default_p=nothing)
                        l_f = NeuralPDE.@RuntimeGeneratedFunction(l_expr)
                        push!(lb_, l_f)
                    end
                end
                for u_ in ub
                    if u_ isa Number
                        push!(ub_, u_)
                    else
                        u_expr = NeuralPDE.build_symbolic_loss_function(pinnrep, nothing;
                                                                        integrand=NeuralPDE._dot_(u_),
                                                                        integrating_depvars=integrating_depvars,
                                                                        param_estim=false,
                                                                        default_p=nothing)
                        u_f = NeuralPDE.@RuntimeGeneratedFunction(u_expr)
                        push!(ub_, u_f)
                    end
                end

                integrand_func = NeuralPDE.@RuntimeGeneratedFunction(integrand)
                ex.args = [
                    :($(Expr(:$, :integral))),
                    :u,
                    Symbol(:cord, num_depvar[1]),
                    :phi,
                    integrating_var_id,
                    integrand_func,
                    lb_,
                    ub_,
                    :($θ),
                ]
                break
            end
        else
            ex.args[i] = _transform_expression(pinnrep, ex.args[i]; is_integral=is_integral,
                                               dict_transformation_vars=dict_transformation_vars,
                                               transformation_vars=transformation_vars)
        end
    end
    return ex
end
