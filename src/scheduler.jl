struct Scheduler{O, S}
    opt::O
    schedule::S
end

Optimisers.init(o::Scheduler, x::AbstractArray) = (t=1, opt=Optimisers.init(o.opt, x))

function Optimisers.apply!(o::Scheduler, state, x, dx)
    eta = o.schedule(state.t)
    opt = Optimisers.adjust(o.opt, eta)
    new_state, new_dx = Optimisers.apply!(opt, state.opt, x, dx)
    return (t=state.t + 1, opt=new_state), new_dx
end

get_opt(s::Scheduler, state) = Optimisers.adjust(s.opt, s.schedule(state.t))

function SciMLBase.__solve(prob::OptimizationProblem, opt::Scheduler,
                           data=Optimization.DEFAULT_DATA; maxiters::Number=0,
                           callback=(args...) -> (false), progress=false, save_best=true,
                           kwargs...)
    if data != Optimization.DEFAULT_DATA
        maxiters = length(data)
    else
        maxiters = Optimization._check_and_convert_maxiters(maxiters)
        data = Optimization.take(data, maxiters)
    end

    θ = copy(prob.u0)
    G = copy(θ)

    local x, min_err, min_θ
    min_err = typemax(eltype(prob.u0)) #dummy variables
    min_opt = 1
    min_θ = prob.u0

    f = Optimization.instantiate_function(prob.f, prob.u0, prob.f.adtype, prob.p)
    state = Optimisers.setup(opt, θ)

    t0 = time()
    Optimization.@withprogress progress name="Training" begin for (i, d) in enumerate(data)
        f.grad(G, θ, d...)
        x = f.f(θ, prob.p, d...)
        cb_call = callback(θ, x...)
        if !(typeof(cb_call) <: Bool)
            error("The callback should return a boolean `halt` for whether to stop the optimization process. Please see the sciml_train documentation for information.")
        elseif cb_call
            break
        end
        msg = OptimizationOptimisers.@sprintf("loss: %.3g", x[1])
        progress && Optimization.ProgressLogging.@logprogress msg i/maxiters

        if save_best
            if first(x) < first(min_err)  #found a better solution
                min_opt = opt
                min_err = x
                min_θ = copy(θ)
            end
            if i == maxiters  #Last iteration, revert to best.
                opt = min_opt
                x = min_err
                θ = min_θ
                callback(θ, x...)
                break
            end
        end
        state, θ = Optimisers.update(state, θ, G)
    end end

    t1 = time()

    return SciMLBase.build_solution(prob, opt, θ, x[1])
    # here should be build_solution to create the output message
end
