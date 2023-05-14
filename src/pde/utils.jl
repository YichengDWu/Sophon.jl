function isongpu(nt::NamedTuple)
    return any(x -> x isa AbstractGPUArray, Lux.fcollect(nt))
end

float32 = Base.Fix1(convert, AbstractArray{Float32})

function get_l2_loss_function(loss_function, dataset)
    loss(θ) = mean(abs2, loss_function(dataset, θ))
    return loss
end

@inline null_additional_loss(phi, θ) = 0

"""
This function is only used for the first order derivative.
"""
forwarddiff(phi, t, εs, order, θ) = ForwardDiff.gradient(sum ∘ Base.Fix2(phi, θ), t)

function taylordiff(phi, x::AbstractMatrix, θ, εs_dnv, direction::NTuple{M, StaticInt{N}}) where {M, N}
    ε = ChainRulesCore.@ignore_derivatives first(εs_dnv)
    return taylordiff(phi, x, θ, ε, Val(M))
end

function taylordiff(phi, x::AbstractMatrix, θ, εs_dnv, direction::Tuple{NTuple{M1, StaticInt{N1}}, NTuple{M2,StaticInt{N2}}}) where {M1,N1,M2,N2}
    ε1 = ChainRulesCore.@ignore_derivatives first(εs_dnv)
    ε2 = ChainRulesCore.@ignore_derivatives last(εs_dnv)
    return let phi=phi, ε1=ε1, M1=M1
        taylordiff((x,θ)->taylordiff(phi, x, θ, ε1, Val(M1)), x, θ, ε2, Val(M2))
    end
end

for N in 1:5
    @eval @inline function taylordiff(phi, x::AbstractVector, θ, ε::AbstractVector,
                                      ::Val{$N})
        return let phi = phi, θ = θ
            TaylorDiff.derivative(i->first(phi(i,θ)), x, ε, Val{$(N+1)}())
        end
    end

    @eval @inline function taylordiff(phi, x::AbstractMatrix, θ, ε::AbstractVector,
                                      ::Val{$N})
        return let phi = phi, θ = θ
            TaylorDiff.derivative(i->phi(i,θ), x, ε, Val{$(N+1)}())
        end
    end
end

@inline function TaylorDiff.derivative(f, x::V1, l::V2,
                            vN::Val{N}) where {V1 <: AbstractVector{<:Number},
                                               V2 <: AbstractVector{<:Number}, N}
    t = map((t0, t1) -> TaylorDiff.make_taylor(t0, t1, vN), x, l)
    return TaylorDiff.extract_derivative(f(t), N)
end

@inline function TaylorDiff.derivative(f, x::V1, l::V2,
                            vN::Val{N}) where {V1 <: AbstractMatrix{<:Number},
                                               V2 <: AbstractVector{<:Number}, N}
    t = broadcast((t0, t1) -> TaylorDiff.make_taylor(t0, t1, vN), x, l)
    o = f(t) # (1,N) matrix
    return map(Base.Fix2(TaylorDiff.extract_derivative, N), o)
end

function ChainRulesCore.rrule(::typeof(*), A::AbstractMatrix{S},
                              t::AbstractMatrix{TaylorScalar{T, N}}) where {N, S <: Number, T}
    project_A = ProjectTo(A)
    project_t = ProjectTo(t)
    function gemv_pullback(x̄)
        X̄ = unthunk(x̄)
        dA = @thunk(project_A(broadcast(axes(X̄, 1), axes(t, 1)') do i, j
            mapreduce(contract, +, @view(X̄[i, :]), @view(t[j, :]))
        end))
        dB = @thunk(project_t(transpose(A)*X̄))
        NoTangent(), dA, dB
    end
    return A * t, gemv_pullback
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
