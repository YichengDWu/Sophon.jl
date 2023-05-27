function isongpu(nt::NamedTuple)
    return any(x -> x isa AbstractGPUArray, Lux.fcollect(nt))
end

function get_l2_loss_function(loss_function, dataset)
    loss(θ) = mean(abs2, loss_function(dataset, θ))
    return loss
end

@inline null_additional_loss(phi, θ) = 0

"""
This function is only used for the first order derivative.
"""
forwarddiff(phi, t, εs, order, θ) = ForwardDiff.gradient(sum ∘ Base.Fix2(phi, θ), t)

@inline function finitediff(phi, x, θ, ε::AbstractVector{T}, h::T, ::Val{1}) where {T<:AbstractFloat}
    ε = ChainRulesCore.@ignore_derivatives adapt(parameterless_type(ComponentArrays.getdata(θ)), ε)
    return (phi(x .+ ε, θ) .- phi(x .- ε, θ)) .* (h / 2)
end

@inline function finitediff(phi, x, θ, ε::AbstractVector{T}, h::T, ::Val{2}) where {T<:AbstractFloat}
    ε = ChainRulesCore.@ignore_derivatives adapt(parameterless_type(ComponentArrays.getdata(θ)), ε)
    return (phi(x .+ ε, θ) .+ phi(x .- ε, θ) .- 2 .* phi(x, θ)) .* h^2
end

@inline function finitediff(phi, x, θ, ε::AbstractVector{T}, h::T, ::Val{3}) where {T<:AbstractFloat}
    ε = ChainRulesCore.@ignore_derivatives adapt(parameterless_type(ComponentArrays.getdata(θ)), ε)
    return (phi(x .+ 2 .* ε, θ) .- 2 .* phi(x .+ ε, θ) .+ 2 .* phi(x .- ε, θ) -
            phi(x .- 2 .* ε, θ)) .* h^3 ./ 2
end

@inline function finitediff(phi, x, θ, ε::AbstractVector{T}, h::T, ::Val{4}) where {T<:AbstractFloat}
    ε = ChainRulesCore.@ignore_derivatives adapt(parameterless_type(ComponentArrays.getdata(θ)), ε)
    return (phi(x .+ 2 .* ε, θ) .- 4 .* phi(x .+ ε, θ) .+ 6 .* phi(x, θ) .-
            4 .* phi(x .- ε, θ) .+ phi(x .- 2 .* ε, θ)) .* h^4
end

function finitediff(phi, x, θ, dim::Int, order::Int)
    ε, h = ChainRulesCore.@ignore_derivatives get_ε_h(finitediff, size(x, 1), dim, eltype(θ), order)

    ε = adapt(parameterless_type(ComponentArrays.getdata(θ)), ε)

    if order == 4
        return (phi(x .+ 2 .* ε, θ) .- 4 .* phi(x .+ ε, θ) .+ 6 .* phi(x, θ) .-
                4 .* phi(x .- ε, θ) .+ phi(x .- 2 .* ε, θ)) .* h^4
    elseif order == 3
        return (phi(x .+ 2 .* ε, θ) .- 2 .* phi(x .+ ε, θ, phi) .+ 2 .* phi(x .- ε, θ) -
                phi(x .- 2 .* ε, θ)) .* h^3 ./ 2
    elseif order == 2
        return (phi(x .+ ε, θ) .+ phi(x .- ε, θ) .- 2 .* phi(x, θ)) .* h^2
    elseif order == 1
        return (phi(x .+ ε, θ) .- phi(x .- ε, θ)) .* h ./ 2
    else
        error("The order $order is not supported!")
    end
end

# only order = 1 is supported
function upwind(phi, x, θ, ε::AbstractVector{T}, h::T, ::Val{1}) where {T<:AbstractFloat}
    ε = ChainRulesCore.@ignore_derivatives adapt(parameterless_type(ComponentArrays.getdata(θ)), ε)
    return (3 .* phi(x, θ) .- 4 .* phi(x .- ε, θ) .+ phi(x .- 2 .* ε, θ)) .* (h / 2)
end

function get_ε_h(::typeof(finitediff), dim, der_num, fdtype, order)
    epsilon = ^(eps(fdtype), one(fdtype) / (2 + order))
    ε = zeros(fdtype, dim)
    ε[der_num] = epsilon
    return ε, inv(epsilon)
end

function get_ε_h(::typeof(upwind), dim, der_num, fdtype, order)
    epsilon = ^(eps(fdtype), one(fdtype) / (2 + order))
    ε = zeros(fdtype, dim)
    ε[der_num] = epsilon
    return ε, inv(epsilon)
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

function insert_row(matrix, vector, index)
    if index > size(matrix, 1) + 1 || index < 1
        error("Index out of bounds.")
    end

    if size(matrix, 2) != size(vector, 2)
        error("Dimensions mismatch.")
    end

    return vcat(@view(matrix[1:index-1, :]), vector, @view(matrix[index:end, :]))
end
