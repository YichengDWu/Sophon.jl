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
                  dname => [Symbol(ModelingToolkit.value(argument))
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

_beats(x, y::Nothing) = true
_beats(x::Number, y::Number) = true
_beats(x::Number, y::Symbol) = false
_beats(x::Symbol, y::Number) = true
_beats(x::Symbol, y::Symbol) = false

function get_arguments(eq::ModelingToolkit.Equation, dict_indvars, dict_depvar_input)
    args = Dict{Symbol,Any}(keys(dict_indvars) .=> nothing)
    expr = ModelingToolkit.toexpr(eq.lhs - eq.rhs)

    function update_args!(args, key, val)
        old_val = get(args, key, nothing)
        if _beats(val, old_val)
            args[key] = val
        end
    end
    postwalk(expr) do x
        if @capture(x, gg_(xs__))
            if gg ∈ keys(dict_depvar_input)
                map(dict_depvar_input[gg], xs) do key, arg
                    key ∈ keys(args) && update_args!(args, key, arg)
                end
            end
        end
        x
    end
    values(sort(args; by=x->dict_indvars[x])) |> collect
end

function get_arguments(eq::Vector{<:ModelingToolkit.Equation}, dict_indvars, dict_depvar_input)
    map(eq) do eq
        get_arguments(eq, dict_indvars, dict_depvar_input)
    end
end

function get_bounds(domains::Vector{Symbolics.VarDomainPairing}, eqs, bcs, eltypeθ,
                    dict_indvars, dict_depvar_input)
    dict_span = Dict([Symbol(d.variables) => [infimum(d.domain), supremum(d.domain)]
                      for d in domains])
    pde_args = get_arguments(eqs, dict_indvars, dict_depvar_input)
    pde_bounds = map(pde_args) do pde_arg
        bds = mapreduce(s -> get(dict_span, s, fill(s, 2)), hcat, pde_arg)
        bds = eltypeθ.(bds)
        return bds[1, :], bds[2, :]
    end

    bound_args = get_arguments(bcs, dict_indvars, dict_depvar_input)
    bcs_bounds = map(bound_args) do bound_arg
        bds = mapreduce(s -> get(dict_span, s, fill(s, 2)), hcat, bound_arg)
        bds = eltypeθ.(bds)
        return bds[1, :], bds[2, :]
    end
    return pde_bounds, bcs_bounds
end

function get_bounds(pde::ModelingToolkit.PDESystem)
    (; eqs, bcs, domain, ivs, dvs) = pde
    _, _, dict_indvars, _, dict_depvar_input = get_vars(ivs, dvs)
    bounds = get_bounds(domain, eqs, bcs, Float64, dict_indvars, dict_depvar_input)
    return bounds
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


function build_symbolic_loss_function(pinnrep::NamedTuple,
                                      eq::Symbolics.Equation)
    (; indvars, depvars, dict_depvar_input, derivative, multioutput, dict_indvars) = pinnrep

    vars = :(coord, θ)
    ex = Expr(:block)

    # Step 1: interpolate phi and derivative in the expression
    phi = pinnrep.phi
    push!(ex.args, Expr(:(=), :phi, phi))
    push!(ex.args, Expr(:(=), :derivative, derivative))

    # Step 2: assign phi and θ to each depvar
    if multioutput
        push!(ex.args, Expr(:(=),
                            Expr(:tuple, [Symbol(:phi_, i) for i in depvars]...),
                            Expr(:tuple, [:(phi.$(i)) for i in depvars]...)))
        push!(ex.args, Expr(:(=),
                            Expr(:tuple, [Symbol(:θ_, i) for i in depvars]...),
                            Expr(:tuple, [:(θ.$(i)) for i in depvars]...)))
    else
        push!(ex.args, Expr(:(=), Symbol(:phi_, depvars[1]), :phi))
        push!(ex.args, Expr(:(=), Symbol(:θ_, depvars[1]), :θ))
    end

    # Step 3: split coord into all indvars
    this_eq_expr = eq_to_expr(eq)
    this_eq_depvars = Symbol[]
    this_eq_indvars = Symbol[]
    postwalk(this_eq_expr) do x
        if @capture(x, f_(args__))
            if f in depvars
                push!(this_eq_depvars, f)
            end
            push!(this_eq_indvars, filter(Base.Fix2(in, indvars), args)...)
        end
        return x
    end
    this_eq_depvars = unique(this_eq_depvars)
    this_eq_indvars = unique(this_eq_indvars)

    let_block = Expr(:let)
    push!(let_block.args, Expr(:(=),
                               Expr(:tuple, this_eq_indvars...),
                               Expr(:tuple, [:(coord[[$(dict_indvars[x])], :]) for x in this_eq_indvars]...)))

    # Step 4: build coord for each depvar
    coord_block = Expr(:block)
    for d in this_eq_depvars
        coord_expr = if length(dict_depvar_input[d]) == length(indvars)
                        :coord
                     elseif length(dict_depvar_input[d]) == 1
                        :($(dict_depvar_input[d][1]))
                     else
                        :(vcat($(dict_depvar_input[d]...)))
                     end
        push!(coord_block.args, Expr(:(=),
                                   Symbol(:coord_, d),
                                   coord_expr))
    end

    # Step 5: build loss function for this equation
    loss_function = expr_to_residual_function(pinnrep, this_eq_expr)
    push!(let_block.args, Expr(:block, coord_block, loss_function))

    push!(ex.args, let_block)
    return vars, ex
end

function build_loss_function(pinnrep::NamedTuple, eq::Symbolics.Equation, i)
    vars, ex = build_symbolic_loss_function(pinnrep, eq)
    expr = Expr(:function,
                Expr(:call, Symbol(:residual_function_, i), vars.args[1], vars.args[2]), ex)
    return eval(expr)
end

function eq_to_expr(eq)
    eq = eq.lhs - eq.rhs
    #eq = ModelingToolkit.expand_derivatives(eq)
    return ModelingToolkit.toexpr(eq)
end

function expr_to_residual_function(pinnrep::NamedTuple, expr::Expr)
    # turn :($(Differential(t))) into :(Differential(t))
    expr = postwalk(expr) do x
        if @capture(x, f_(xs__))
            if f isa Differential
                :(Differential($(Symbol(f.x)))($(xs...)))
            else
                :($f($(xs...)))
            end
        else
            x
        end
    end

    expr = transform_expression(pinnrep, expr)
    return expr
end

const mixed_derivative_patterns = (
    ((1,1), :((Differential(dr1_))((Differential(dr2_))(ff_(args__))))),
    ((2,1), :((Differential(dr1_))((Differential(dr1_))((Differential(dr2_))(ff_(args__)))))),
    ((2,2), :((Differential(dr1_))((Differential(dr1_))((Differential(dr2_))((Differential(dr2_))(ff_(args__))))))),
)

const derivative_patterns = (
    (1,:((Differential(dr_))(ff_(args__)))),
    (2,:((Differential(dr_))((Differential(dr_))(ff_(args__))))),
    (3,:((Differential(dr_))((Differential(dr_))((Differential(dr_))(ff_(args__)))))),
    (4,:((Differential(dr_))((Differential(dr_))((Differential(dr_))((Differential(dr_))(ff_(args__)))))))
)

function transform_expression(pinnrep::NamedTuple{names}, ex::Expr) where {names}
    (; indvars, dict_depvars, dict_depvar_input, fdtype, init_params, derivative) = pinnrep
    use_gpu = isongpu(init_params)

    # Step 1: Replace all the derivatives with the derivative function
    ex = prewalk(ex) do x
        quoted_x = Meta.quot(x)

        for ((order1, order2), pattern) in reverse(mixed_derivative_patterns)
            if @eval @capture($quoted_x, $pattern) && dr1 !== dr2
                ε1, h1 = get_ε_h(derivative, length(args), findfirst(==(dr1), dict_depvar_input[ff]), fdtype, order1)
                ε2, h2 = get_ε_h(derivative, length(args), findfirst(==(dr2), dict_depvar_input[ff]), fdtype, order2)
                ε1 = use_gpu ? adapt(CuArray, ε1) : ε1
                ε2 = use_gpu ? adapt(CuArray, ε2) : ε2

                return :(derivative((x,ps)->derivative(phi_u, x, ps, $ε2, $h2, $(Val(order2))),
                                     coord_u, θ, $ε1, $h1, $(Val(order1))))
            end
        end

        for (order, pattern) in reverse(derivative_patterns)
            if @eval @capture($quoted_x, $pattern)
                ε, h = get_ε_h(derivative, length(args), findfirst(==(dr), dict_depvar_input[ff]), fdtype, order)
                ε = use_gpu ? adapt(CuArray, ε) : ε
                return :(derivative($(Symbol(:phi, :_, ff)), $(Symbol(:coord, :_, ff)), $(Symbol(:θ, :_, ff)), $ε, $h, $(Val(order))))
            end
        end
        return x
    end

    # Step 2: Convert u(x,t) to phi_u(coord_u, θ_u), u(x, 1.0) to phi_u(vcat(x, zero(view(coord,[1],:)) .+ 1.0), θ_u)
    # Step 3: convert sin(x,t) to sin.(x,t)
    ex = postwalk(ex) do x
        if @capture(x, gg_(xs__))
            if gg in keys(dict_depvars)
                if xs == indvars
                    return :($(Symbol(:phi, :_, gg))($(Symbol(:coord, :_, gg)), $(Symbol(:θ, :_, gg))))
                elseif hasproperty(xs[1], :head) && xs[1].head === :call
                    return :($(Symbol(:phi, :_, gg))($(xs...), $(Symbol(:θ, :_, gg))))
                else
                    cs = map(xs) do i
                        i isa Symbol ? i : :(zero(view(coord, [1], :)) .+ $i)
                    end
                    return :($(Symbol(:phi, :_, gg))(vcat($(cs...)), $(Symbol(:θ, :_, gg))))
                end
            elseif gg===:derivative
                return:($gg($(xs...)))
            else
                return :($gg.($(xs...)))
            end
        else
            return x
        end
    end
end


function get_where_t_is(pde_system)
    (; indvars, depvars) = pde_system
    depvars, indvars, dict_indvars, dict_depvars, dict_depvar_input = get_vars(indvars,
                                                                               depvars)
    return  dict_indvars[:t]
end
