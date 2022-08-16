function Base.fill!(A::ComponentArrays.GPUComponentArray{T}, x) where {T}
    length(A) == 0 && return A
    GPUArrays.gpu_call(A, convert(T, x)) do ctx, a, val
        idx = GPUArrays.@linearidx(a)
        @inbounds a[idx] = val
        return
    end
    A
end

LinearAlgebra.dot(x::ComponentArrays.GPUComponentArray, y::ComponentArrays.GPUComponentArray) =
    dot(getdata(x), getdata(y))
LinearAlgebra.norm(ca::ComponentArrays.GPUComponentArray, p::Real) = norm(getdata(ca), p)
LinearAlgebra.rmul!(ca::ComponentArrays.GPUComponentArray, b::Number) = GPUArrays.generic_rmul!(ca, b)
