module SophonTaylorDiffExt

if isdefined(Base, :get_extension)
    using TaylorDiff
    import TaylorDiff: derivative, make_taylor, raise, extract_derivative, value
else
    using ..TaylorDiff
    import ..TaylorDiff: derivative, make_taylor, raise, extract_derivative, value
end

using Sophon
import Sophon.ChainRulesCore as CRC
import NNlib
import LinearAlgebra

function NNlib.tanh_fast(t::TaylorScalar{T,2}) where {T}
    t0, t1 = value(t)
    return TaylorScalar{T,2}(CRC.frule((CRC.NoTangent(), t1), NNlib.tanh_fast, t0))
end
function NNlib.tanh_fast(t::TaylorScalar{T,N}) where {T,N}
    t1 = TaylorScalar{T,N-1}(t)
    df = 1 - NNlib.tanh_fast(t1)^2
    return raise(NNlib.tanh_fast(value(t)[1]), df, t)
end

function CRC.rrule(::typeof(*), A::AbstractMatrix{S},
               t::AbstractVector{TaylorScalar{T, N}}) where {N, S <: Number, T}
    project_A = CRC.ProjectTo(A)
    function gemv_pullback(x̄)
        x̂ = reinterpret(reshape, T, x̄)
        t̂ = reinterpret(reshape, T, t)
        CRC.NoTangent(), CRC.@thunk(project_A(transpose(x̂) * t̂)), CRC.@thunk(transpose(A)*x̄)
    end
    return A * t, gemv_pullback
end

function CRC.rrule(::typeof(*), A::AbstractMatrix{S},
               t::AbstractMatrix{TaylorScalar{T,N}}) where {N, S <: Number, T}
    project_t = CRC.ProjectTo(t)
    function gemv_pullback(x̄)
        X̄ = CRC.unthunk(x̄)
        X̂ = reinterpret(reshape, T, X̄)
        T̂ = reinterpret(reshape, T, t)

        dA = CRC.@thunk begin
                C = zero(A)
                @inbounds for n in axes(X̂, 3)
                    LinearAlgebra.mul!(C, transpose(@view(X̂[:, :, n])), @view(T̂[:, :, n]), true, true)
                end
                C
            end
        dB = CRC.@thunk(project_t(transpose(A)*X̄))
        CRC.NoTangent(), dA, dB
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

@inline extract_derivative(t::TaylorScalar, ::Val{N}) where {N} = value(t)[N]
# batched version
@inline function derivative(f, x::AbstractMatrix{T}, l::AbstractVector{T},
                            vN::Val{N}) where {T <: Number, N}
    t = broadcast((t0, t1) -> TaylorDiff.make_taylor(t0, t1, vN), x, l)
    return map(Base.Fix2(extract_derivative, vN), f(t))
end

@inline function taylordiff(phi, x, θ, ε::AbstractVector{T}, h::T, ::Val{N}) where {T <: Number, N}
    return TaylorDiff.derivative(Base.Fix2(phi, θ), x, ε, Val{N+1}())
end

function Sophon.get_ε_h(::typeof(taylordiff), dim, der_num, fdtype, order)
    epsilon = one(fdtype)
    ε = zeros(fdtype, dim)
    ε[der_num] = epsilon
    return ε, epsilon
end

function __init__()
    @static if VERSION >= v"1.9.0"
        setproperty!(Sophon, :taylordiff, taylordiff)
    end
end

end
