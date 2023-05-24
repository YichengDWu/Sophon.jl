module SophonTaylorDiffExt

if isdefined(Base, :get_extension)
    using TaylorDiff, NNlib, ChainRulesCore
    import NNlib: tanh_fast
    import TaylorDiff: derivative, make_taylor, raise, extract_derivative
    import ChainRulesCore: rrule
else
    using ..TaylorDiff, ..NNlib, ..ChainRulesCore
    import ..NNlib: tanh_fast
    import ..TaylorDiff: derivative, make_taylor, raise, extract_derivative
    import ..ChainRulesCore: rrule
end

function tanh_fast(t::TaylorScalar{T,2}) where {T}
    t0, t1 = TaylorDiff.value(t)
    return TaylorScalar{T,2}(frule((NoTangent(), t1), tanh_fast, t0))
end
function tanh_fast(t::TaylorScalar{T,N}) where {T,N}
    t1 = TaylorScalar{T,N-1}(t)
    df = 1 - tanh_fast(t1)^2
    return raise(tanh_fast(TaylorDiff.value(t)[1]), df, t)
end

function rrule(::typeof(*), A::AbstractMatrix{S},
               t::AbstractVector{TaylorScalar{T, N}}) where {N, S <: Number, T}
    project_A = ProjectTo(A)
    function gemv_pullback(x̄)
        x̂ = reinterpret(reshape, T, x̄)
        t̂ = reinterpret(reshape, T, t)
        NoTangent(), @thunk(project_A(transpose(x̂) * t̂)), @thunk(transpose(A)*x̄)
    end
    return A * t, gemv_pullback
end

function rrule(::typeof(*), A::AbstractMatrix{S},
               t::AbstractMatrix{TaylorScalar{T, N}}) where {N, S <: Number, T}
    project_A = ProjectTo(A)
    function gemv_pullback(x̄)
        x̂ = reinterpret(T, x̄)
        t̂ = reinterpret(T, t)
        NoTangent(), @thunk(project_A(transpose(x̂) * t̂)), @thunk(transpose(A)*x̄)
    end
    return A * t, gemv_pullback
end

@inline function derivative(f, x::AbstractVector{T}, l::AbstractVector{T},
                            order::Int64) where {T <: Number}
    derivative(f, x, l, Val{order + 1}())
end

@inline function derivative(f, x::AbstractVector{T}, l::AbstractVector{T},
                            vN::Val{N}) where {T <: Number, N}
    t = broadcast((t0, t1) -> make_taylor(t0, t1, vN), x, l)
    return extract_derivative(f(t), N)
end

end
