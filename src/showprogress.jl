using MacroTools
using ProgressBars
using Printf

macro showprogress(expr)
    @capture(expr, res_ = Optimization.solve(prob_, method_; kwargs__)) ||
        error("Unsupported expression format")

    maxiters = nothing
    callback = nothing

    for kw in kwargs
        isnothing(maxiters) && @capture(kw, maxiters=maxiters_)
        isnothing(callback) && @capture(kw, callback=callback_)
    end

    if maxiters === nothing
        error("maxiters is not specified")
    end

    iter_var = gensym(:iter)
    progress_expr = :($iter_var = ProgressBars.ProgressBar(1:$maxiters))

    if isnothing(callback)
        callback_var = gensym(:callback)
        callback_def = quote
            function $callback_var(p, l)
                ProgressBars.update($iter_var)
                ProgressBars.set_description($iter_var, string(Printf.@sprintf("%.6e", l)))
                return false
            end
        end
    else
        callback_var = gensym(:callback)
        callback_def = quote
            function $callback_var(p, l)
                update($iter_var)
                set_description($iter_var, string(@sprintf("%.6e", l)))
                return $callback(p, l)
            end
        end
    end

    new_func_call = Expr(:call, :(Optimization.solve), esc(prob), esc(method), kwargs..., :($(Expr(:kw, :callback, callback_var))))
    assignment_expr = Expr(:(=), esc(res), new_func_call)

    return quote
        $progress_expr
        $callback_def
        $assignment_expr
    end
end
