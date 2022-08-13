struct Scheduler{O, S}
    opt::O
    schedule::S
end

Optimisers.init(o::Scheduler, x::AbstractArray) =
    (t = 1, opt = Optimisers.init(o.opt, x))

function Optimisers.apply!(o::Scheduler, state, x, dx)
    eta = o.schedule(state.t)
    opt = Optimisers.adjust(o.opt, eta)
    new_state, new_dx = Optimisers.apply!(opt, state.opt, x, dx)
    return (t = state.t + 1, opt = new_state), new_dx
end

get_opt(s::Scheduler, state) = Optimisers.adjust(s.opt, s.schedule(state.t))
