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

for N in 1:5
    @eval begin
        $(Symbol(:broadcasted_make_taylor_, N))(t0,t1) = CRC.@ignore_derivatives broadcast((t0, t1) -> make_taylor(t0, t1, $(Val(N))), t0, t1)

        function CRC.rrule(f::typeof($(Symbol(:broadcasted_make_taylor_, N))), x::AbstractVector, y::AbstractVector)
            o = f(x, y)
            function f_pullback(x̄::AbstractVector{<:TaylorScalar{T}}) where {T}
                x = reinterpret(reshape, T, x̄)
                return CRC.NoTangent(), x[1, :], x[2, :]
            end
            return o, f_pullback
        end

        function CRC.rrule(f::typeof($(Symbol(:broadcasted_make_taylor_, N))), x::AbstractMatrix, y::AbstractVector)
            o = f(x, y)
            function broadcasted_make_taylor_pullback(x̄::AbstractMatrix{<:TaylorScalar{T}}) where {T}
                x = reinterpret(reshape, T, x̄)
                return CRC.NoTangent(), x[1, :, :], x[2, :, 1]
            end
            return o, broadcasted_make_taylor_pullback
        end

        $(Symbol(:broadcasted_extract_derivative_, N))(t) = CRC.@ignore_derivatives map(Base.Fix2(extract_derivative, $(Val(N))), t)

        function CRC.rrule(f::typeof($(Symbol(:broadcasted_extract_derivative_, N))), t::AbstractArray{TaylorScalar{T, L}}) where {T, L}
            function broadcasted_extract_derivative_pullback(x̂)
                Δ = broadcast(x̂) do d
                    TaylorScalar{T, L}(ntuple(j -> j === $N ? d : zero(T), Val{L}()))
                end
                return CRC.NoTangent(), Δ
            end
            return f(t), broadcasted_extract_derivative_pullback
        end
    end
end

@inline function derivative(f, x::AbstractVector{T}, l::AbstractVector{T},
                            order::Int64) where {T <: Number}
    derivative(f, x, l, Val{order + 1}())
end

for N in 1:5
    @eval @inline function derivative(f, x::AbstractVector{T}, l::AbstractVector{T},
                                      ::Val{$N}) where {T <: Number}
        t = $(Symbol(:broadcasted_make_taylor_, N))(x, l)
        return extract_derivative(f(t), N)
    end
end

@inline extract_derivative(t::TaylorScalar, ::Val{N}) where {N} = value(t)[N]
# batched version
for N in 1:5
    @eval @inline function derivative(f, x::AbstractMatrix{T}, l::AbstractVector{T},
                                      ::Val{$N}) where {T <: Number}
        t = $(Symbol(:broadcasted_make_taylor_, N))(x, l)
        return $(Symbol(:broadcasted_extract_derivative_, N))(f(t))
    end
end

@inline function taylordiff(phi, x, θ, ε_::AbstractVector{T}, h::T, ::Val{N}) where {T <: Number, N}
    ε = CRC.@ignore_derivatives convert(Sophon.parameterless_type(x), ε_)
    return TaylorDiff.derivative(Base.Fix2(phi, θ), x, ε, Val{N+1}())
end

function generate_ε(::typeof(taylordiff), dim, der_num, fdtype, order)
    epsilon = one(fdtype)
    ε = zeros(fdtype, dim)
    ε[der_num] = epsilon
    return Sophon.SVector{dim}(ε)
end

for order in 1:4
    for fdtype in (Float32, Float64)
        @eval Sophon.get_h(::typeof(taylordiff), ::Type{$fdtype}, ::Val{$order}) = $(one(fdtype))
    end
end

for l in 1:4
    for d in 1:l
        for order in 1:4
            for fdtype in (Float32, Float64)
                @eval const $(Symbol(:taylordiff_ε, :_, l, :_, d, :_, order, :_, fdtype)) =
                    $(generate_ε(taylordiff, l, d, fdtype, order))

                @eval function Sophon.get_ε(::typeof(taylordiff), ::Val{$l}, ::Val{$d}, ::Type{$fdtype}, ::Val{$order})
                    return $(Symbol(:taylordiff_ε, :_, l, :_, d, :_, order, :_, fdtype))
                end
            end
        end
    end
end

function __init__()
    @static if VERSION >= v"1.9.0"
        setproperty!(Sophon, :taylordiff, taylordiff)
    end
end

end
