const AbstractGPUComponentArray{T,N,A,Ax} = ComponentArray{T,N,A<:GPUArraysCore.AbstractGPUArray,Ax}
const AbstractGPUComponentVector{T,A,Ax} = ComponentVector{T,A<:GPUArraysCore.AbstractGPUVector,Ax}
const AbstractGPUComponentMatrix{T,A,AX} = ComponentMatrix{T,A<:GPUArraysCore.AbstractGPUMatrix,Ax}
const AbstractGPUComponentVecorMat{T} = Union{AbstractGPUComponentVector{T}, AbstractGPUComponentMatrix{T}}

function Base.fill!(A::AbstractGPUComponentArray, x)
    length(A) == 0 && return A
    GPUArrays.gpu_call(A, convert(T, x)) do ctx, a, val
        idx = GPUArrays.@linearidx(a)
        @inbounds a[idx] = val
        return
    end
    return A
end

LinearAlgebra.dot(x::myGPUComponentArray, y::myGPUComponentArray) = dot(getdata(x), getdata(y))
LinearAlgebra.norm(ca::myGPUComponentArray, p::Real) = norm(getdata(ca), p)

function LinearAlgebra.rmul!(ca::myGPUComponentArray, b::Number)
    return GPUArrays.generic_rmul!(ca, b)
end

# This is a workaround and should be removed when I have a better understanding of how to do this.
Base.adjoint(A::ComponentArray{T,N,AA,Ax}) where {T,N,AA<:GPUArrays.AbstractGPUArray,Ax} = adjoint(getdata(A))

LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::AbstractGPUComponentVecorMat, B::AbstractGPUComponentVecorMat, a::Number, b::Number) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::AbstractGPUComponentVecorMat, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Number, b::Number) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::AbstractGPUComponentVecorMat, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, a::Number, b::Number) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, B::AbstractGPUComponentVecorMat, a::Number, b::Number) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, B::AbstractGPUComponentVecorMat, a::Number, b::Number) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Number, b::Number) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, a::Number, b::Number) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Number, b::Number) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, a::Number, b::Number) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)

# specificity hacks
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::AbstractGPUComponentVecorMat, B::AbstractGPUComponentVecorMat, a::Real, b::Real) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::AbstractGPUComponentVecorMat, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Real, b::Real) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::AbstractGPUComponentVecorMat, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, a::Real, b::Real) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, B::AbstractGPUComponentVecorMat, a::Real, b::Real) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, B::AbstractGPUComponentVecorMat, a::Real, b::Real) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Real, b::Real) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, a::Real, b::Real) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Real, b::Real) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat, A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat}, a::Real, b::Real) = CUDA.CUBLAS.generic_matmatmul!(C, A, B, a, b)
