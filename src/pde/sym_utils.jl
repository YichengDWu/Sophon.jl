# aggressively rewrite some functionalities from NeuralPDE.jl for stability
using Base.Broadcast

dottable_(x) = Broadcast.dottable(x)
dottable_(x::Function) = true

_dot_(x) = x
function _dot_(x::Expr)
    dotargs = Base.mapany(_dot_, x.args)
    if x.head === :call && dottable_(x.args[1])
        Expr(:., dotargs[1], Expr(:tuple, dotargs[2:end]...))
    elseif x.head === :comparison
        Expr(:comparison,
             (iseven(i) && dottable_(arg) && arg isa Symbol && isoperator(arg) ?
              Symbol('.', arg) : arg for (i, arg) in pairs(dotargs))...)
    elseif x.head === :$
        x.args[1]
    elseif x.head === :let # don't add dots to `let x=...` assignments
        Expr(:let, undot(dotargs[1]), dotargs[2])
    elseif x.head === :for # don't add dots to for x=... assignments
        Expr(:for, undot(dotargs[1]), dotargs[2])
    elseif (x.head === :(=) || x.head === :function || x.head === :macro) &&
           Meta.isexpr(x.args[1], :call) # function or macro definition
        Expr(x.head, x.args[1], dotargs[2])
    elseif x.head === :(<:) || x.head === :(>:)
        tmp = x.head === :(<:) ? :.<: : :.>:
        Expr(:call, tmp, dotargs...)
    else
        head = String(x.head)::String
        if last(head) == '=' && first(head) != '.' || head == "&&" || head == "||"
            Expr(Symbol('.', head), dotargs...)
        else
            Expr(x.head, dotargs...)
        end
    end
end

function get_ε(dim, der_num, fdtype, order)
    epsilon = ^(eps(fdtype), one(fdtype) / (2 + order))
    ε = zeros(fdtype, dim)
    ε[der_num] = epsilon
    return ε
end

function get_limits(domain)
    if domain isa AbstractInterval
        return [leftendpoint(domain)], [rightendpoint(domain)]
    elseif domain isa ProductDomain
        return collect(map(leftendpoint, DomainSets.components(domain))),
               collect(map(rightendpoint, DomainSets.components(domain)))
    end
end

"""
Finds which dependent variables are being used in an equation.
"""
function pair(eq, depvars, dict_depvar_input)
    expr = ModelingToolkit.toexpr(eq)
    pair_ = map(depvars) do depvar
        if !isempty(find_thing_in_expr(expr, depvar))
            depvar => dict_depvar_input[depvar]
        end
    end
    return Dict(filter(p -> p !== nothing, pair_))
end

get_dict_vars(vars) = Dict([Symbol(v) .=> i for (i, v) in enumerate(vars)])

function get_vars(indvars_, depvars_)
    indvars = ModelingToolkit.getname.(indvars_)
    depvars = Symbol[]
    dict_depvar_input = Dict{Symbol, Vector{Symbol}}()
    for d in depvars_
        if ModelingToolkit.value(d) isa ModelingToolkit.Term || ModelingToolkit.value(d) isa ModelingToolkit.BasicSymbolic
            dname = ModelingToolkit.getname(d)
            push!(depvars, dname)
            push!(dict_depvar_input,
                  dname => [nameof(ModelingToolkit.value(argument))
                            for argument in ModelingToolkit.value(d).arguments])
        else
            dname = ModelingToolkit.getname(d)
            push!(depvars, dname)
            push!(dict_depvar_input, dname => indvars) # default to all inputs if not given
        end
    end

    dict_indvars = get_dict_vars(indvars)
    dict_depvars = get_dict_vars(depvars)
    return depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input
end

function find_thing_in_expr(ex::Expr, thing; ans=[])
    if thing in ex.args
        push!(ans, ex)
    end
    for e in ex.args
        if e isa Expr
            if thing in e.args
                push!(ans, e)
            end
            find_thing_in_expr(e, thing; ans=ans)
        end
    end
    return collect(Set(ans))
end

function get_argument(eqs, dict_indvars, dict_depvars)
    exprs = ModelingToolkit.toexpr.(eqs)
    vars = map(exprs) do expr
        _vars = map(depvar -> find_thing_in_expr(expr, depvar), collect(keys(dict_depvars)))
        f_vars = filter(x -> !isempty(x), _vars)
        return map(x -> first(x), f_vars)
    end
    args_ = map(vars) do _vars
        ind_args_ = map(var -> var.args[2:end], _vars)
        syms = Set{Symbol}()
        filter(vcat(ind_args_...)) do ind_arg
            if ind_arg isa Symbol
                if ind_arg ∈ syms
                    false
                else
                    push!(syms, ind_arg)
                    true
                end
            else
                true
            end
        end
    end
    return args_
end

function get_bounds(domains::Vector{Symbolics.VarDomainPairing}, eqs, bcs, eltypeθ,
                    dict_indvars, dict_depvars)
    dict_span = Dict([Symbol(d.variables) => [infimum(d.domain), supremum(d.domain)]
                      for d in domains])
    pde_args = get_argument(eqs, dict_indvars, dict_depvars)
    pde_bounds = map(pde_args) do pde_arg
        bds = mapreduce(s -> get(dict_span, s, fill(s, 2)), hcat, pde_arg)
        bds = eltypeθ.(bds)
        return bds[1, :], bds[2, :]
    end

    bound_args = get_argument(bcs, dict_indvars, dict_depvars)
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

function get_bounds(pde::ModelingToolkit.PDESystem)
    (; eqs, bcs, domain, ivs, dvs) = pde
    _, _, dict_indvars, dict_depvars, _ = get_vars(ivs, dvs)
    bounds = get_bounds(domain, eqs, bcs, Float64, dict_indvars, dict_depvars)
    return bounds
end

function get_variables(eqs, _indvars::Array, _depvars::Array)
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(_indvars,
                                                                               _depvars)
    return get_variables(eqs, dict_indvars, dict_depvars)
end

function get_variables(eqs, dict_indvars, dict_depvars)
    bc_args = get_argument(eqs, dict_indvars, dict_depvars)
    return map(barg -> filter(x -> x isa Symbol, barg), bc_args)
end

"""
[Dx(u1(x,y)) + 4*Dy(u2(x,y)) ~ 0,
 Dx(u2(x,y)) + 9*Dy(u1(x,y)) ~ 0]

:((coord, θ) -> begin
  #= ... =#
  #= ... =#
  begin
      (θ_depvar_1, θ_depvar_2) = (θ.depvar_1, θ.depvar_2)
      (phi_depvar_1, phi_depvar_2) = (phi.depvar_1, phi.depvar_2)
      let (x, y) = (coord[1], coord[2])
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
"""
function build_symbolic_loss_function(pinnrep::NamedTuple{names},
                                      eq::Symbolics.Equation) where {names}
    (; depvars, dict_depvars, dict_depvar_input, derivative, multioutput, dict_indvars) = pinnrep

    loss_function, pos, values = parse_equation(pinnrep, eq)
    this_eq_pair = pair(eq, depvars, dict_depvar_input)
    this_eq_indvars = unique(vcat([getindex(this_eq_pair, v) for v in keys(this_eq_pair)]...))
    this_eq_indvars_locations = Dict([depvar => [findfirst(==(indvar), this_eq_indvars)
                                                 for indvar in this_eq_pair[depvar]]
                                      for depvar in keys(this_eq_pair)])
    maybe_vcat = Dict([depvar => length(this_eq_indvars_locations[depvar]) == length(this_eq_indvars)
                       for depvar in keys(this_eq_pair)])

    vars = :(coord, θ, pfs)
    ex = Expr(:block)
    if :pvs ∈ names
        (; pinn, coord_branch_net) = pinnrep

        push!(ex.args, Expr(:(=), :deeponet, pinn.phi))
        push!(ex.args, Expr(:(=), :derivative, derivative))
        push!(ex.args, Expr(:(=), :coord_branch_net, coord_branch_net))
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
                   :(transpose(get_pfs_output(coord_branch_net...)))))
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

        vars_θ = Expr(:(=), ModelingToolkit.build_expr(:tuple, θs),
                      ModelingToolkit.build_expr(:tuple, expr_θ))
        push!(ex.args, vars_θ)

        vars_phi = Expr(:(=), ModelingToolkit.build_expr(:tuple, phis),
                        ModelingToolkit.build_expr(:tuple, expr_phi))
        push!(ex.args, vars_phi)
    end

    eq_pair_expr = Expr[]
    if pos isa Int
        v = first(keys(this_eq_pair))
        ivars_l = convert(Vector{Any}, deepcopy(this_eq_pair[v]))
        ivars_r = convert(Vector{Any}, deepcopy(this_eq_pair[v]))
        ivars_l[pos] = :(zero(coord[[1], :]) .+ $(values[1]))
        ivars_r[pos] = :(zero(coord[[1], :]) .+ $(values[2]))
        push!(eq_pair_expr, :($(Symbol(:coord, :_, :($v), :_l)) = vcat($(ivars_l...))))
        push!(eq_pair_expr, :($(Symbol(:coord, :_, :($v), :_r)) = vcat($(ivars_r...))))
        this_eq_indvars = this_eq_indvars[setdiff(1:length(this_eq_indvars), pos)]
    else
        for v in keys(this_eq_pair)
            push!(eq_pair_expr,
                  :($(Symbol(:coord, :_, :($v))) = $(ifelse(maybe_vcat[v],
                                                     :coord,
                                                     :(vcat($(this_eq_pair[v]...)))))))
                  #view(coord, $(this_eq_indvars_locations[v]), :)))
        end
    end
    vcat_expr = Expr(:block, :($(eq_pair_expr...)))
    vcat_expr_loss_functions = Expr(:block, vcat_expr, loss_function)

    indvars_ex = [:(coord[[$i], :]) for (i, x) in enumerate(this_eq_indvars)]
    left_arg_pairs, right_arg_pairs = this_eq_indvars, indvars_ex
    vars_eq = Expr(:(=), ModelingToolkit.build_expr(:tuple, left_arg_pairs),
                   ModelingToolkit.build_expr(:tuple, right_arg_pairs))

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

function parse_equation(pinnrep::NamedTuple, eq)
    eq_lhs = isequal(ModelingToolkit.expand_derivatives(eq.lhs), 0) ? eq.lhs :
             ModelingToolkit.expand_derivatives(eq.lhs)
    eq_rhs = isequal(ModelingToolkit.expand_derivatives(eq.rhs), 0) ? eq.rhs :
             ModelingToolkit.expand_derivatives(eq.rhs)
    left_expr = ModelingToolkit.toexpr(eq_lhs)
    right_expr = ModelingToolkit.toexpr(eq_rhs)

    tran_left_expr = transform_expression(pinnrep, ModelingToolkit.toexpr(eq_lhs))
    tran_right_expr = transform_expression(pinnrep, ModelingToolkit.toexpr(eq_rhs))
    dot_left_expr = _dot_(tran_left_expr)
    dot_right_expr = _dot_(tran_right_expr)

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

function is_periodic_bc(bcs, eq, depvars, left_expr::Expr, right_expr::Symbol)
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
                    [:($(Expr(:$, :phi))), Symbol(:coord, :_, e), :θ]
                else
                    [
                        :($(Expr(:$, Symbol(:phi, :_, e)))),
                        Symbol(:coord, :_, e),
                        Symbol(:θ, :_, e),
                    ]
                end
                break
            elseif e in keys(dict_pmdepvars)
                ex.args[1] = :($(Expr(:$, :get_pfs_output)))
                break
            elseif e isa ModelingToolkit.Differential
                derivative_variables = Symbol[]
                order = 0
                while (_args[1] isa ModelingToolkit.Differential)
                    order += 1
                    push!(derivative_variables, ModelingToolkit.toexpr(_args[1].x))
                    _args = _args[2].args # handle Dx(Dy(u)) case order = 2
                end
                depvar = _args[1]
                num_depvar = dict_depvars[depvar]
                indvars = _args[2:end]
                dict_interior_indvars = Dict([indvar .=> j
                                              for (j, indvar) in enumerate(dict_depvar_input[depvar])])
                dim_l = length(dict_interior_indvars)

                εs = [get_ε(dim_l, d, fdtype, order) for d in 1:dim_l]
                undv = [dict_interior_indvars[d_p] for d_p in derivative_variables]
                mixed = any(!=(first(undv)), undv)
                εs_dnv = [εs[d] for d in reverse(undv)]
                epsilon = [inv(first(ε[ε .!= zero(ε)])) for ε in εs_dnv]

                ex.args = if !multioutput
                    [
                        :($(Expr(:$, :derivative))),
                        :phi,
                        Symbol(:coord, :_, depvar),
                        :θ,
                        εs_dnv,
                        epsilon,
                        Val{order}(),
                        Val{mixed}()
                    ]
                else
                    [
                        :($(Expr(:$, :derivative))),
                        Symbol(:phi, :_, depvar),
                        Symbol(:coord, :_, depvar),
                        Symbol(:θ, :_, depvar),
                        εs_dnv,
                        epsilon,
                        Val{order}(),
                        Val{mixed}()
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


function get_where_t_is(pde_system)
    (; indvars, depvars) = pde_system
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(indvars,
                                                                               depvars)
    return  dict_indvars[:t]
end
