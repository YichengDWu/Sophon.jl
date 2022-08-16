function Base.fill!(A::ComponentArrays.ComponentArray{T, N, A, Ax},
                    x) where {T, N, A <: GPUArrays.AbstractGPUArray, Ax}
    length(A) == 0 && return A
    GPUArrays.gpu_call(A, convert(T, x)) do ctx, a, val
        idx = GPUArrays.@linearidx(a)
        @inbounds a[idx] = val
        return
    end
    return A
end

function LinearAlgebra.dot(x::ComponentArrays.ComponentArray{T1, N1, A1, Ax1},
                           y::ComponentArrays.ComponentArray{T2, N2, A2, Ax2}) where {T1,
                                                                                      T2,
                                                                                      N1,
                                                                                      N2,
                                                                                      A1 <:
                                                                                      GPUArrays.AbstractGPUArray,
                                                                                      A2 <:
                                                                                      GPUArrays.AbstractGPUArray,
                                                                                      Ax1,
                                                                                      Ax2}
    return dot(getdata(x), getdata(y))
end
function LinearAlgebra.norm(ca::ComponentArrays.ComponentArray{T, N, A, Ax},
                            p::Real) where {T, N, A <: GPUArrays.AbstractGPUArray, Ax}
    return norm(getdata(ca), p)
end
function LinearAlgebra.rmul!(ca::ComponentArrays.ComponentArray{T, N, A, Ax},
                             b::Number) where {T, N, A <: GPUArrays.AbstractGPUArray, Ax}
    return GPUArrays.generic_rmul!(ca, b)
end
