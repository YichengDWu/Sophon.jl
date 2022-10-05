const AbstractGPUComponentArray{T, N, Ax} = ComponentArray{T, N,
                                                           <:GPUArraysCore.AbstractGPUVector,
                                                           Ax}
const AbstractGPUComponentVector{T, Ax} = ComponentArray{T, 1,
                                                         <:GPUArraysCore.AbstractGPUVector,
                                                         Ax}
const AbstractGPUComponentMatrix{T, Ax} = ComponentArray{T, 2,
                                                         <:GPUArraysCore.AbstractGPUMatrix,
                                                         Ax}
const AbstractGPUComponentVecorMat{T, Ax} = Union{AbstractGPUComponentVector{T, Ax},
                                                  AbstractGPUComponentMatrix{T, Ax}}

function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::AbstractGPUComponentVecorMat,
                            B::AbstractGPUComponentVecorMat, a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::AbstractGPUComponentVecorMat,
                            B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat},
                            a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::AbstractGPUComponentVecorMat,
                            B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUComponentVecorMat},
                            a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end

function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::AbstractGPUComponentVecorMat,
                            B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat},
                            a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::AbstractGPUComponentVecorMat,
                            B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUComponentVecorMat},
                            a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat},
                            B::AbstractGPUComponentVecorMat, a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUComponentVecorMat},
                            B::AbstractGPUComponentVecorMat, a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat},
                            B::AbstractGPUComponentVecorMat, a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUComponentVecorMat},
                            B::AbstractGPUComponentVecorMat, a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat},
                            B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat},
                            a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUComponentVecorMat},
                            B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUComponentVecorMat},
                            a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat},
                            B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat},
                            a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUComponentVecorMat},
                            B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUComponentVecorMat},
                            a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat},
                            B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat},
                            a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUComponentVecorMat},
                            B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUComponentVecorMat},
                            a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat},
                            B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat},
                            a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUComponentVecorMat},
                            B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUComponentVecorMat},
                            a::Number, b::Number)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end

function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::AbstractGPUComponentVecorMat,
                            B::AbstractGPUComponentVecorMat, a::Real, b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::AbstractGPUComponentVecorMat,
                            B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Real,
                            b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::AbstractGPUComponentVecorMat,
                            B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Real,
                            b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::AbstractGPUComponentVecorMat,
                            B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUComponentVecorMat},
                            a::Real, b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat},
                            B::AbstractGPUComponentVecorMat, a::Real, b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUComponentVecorMat},
                            B::AbstractGPUComponentVecorMat, a::Real, b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
ends
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat},
                            B::AbstractGPUComponentVecorMat, a::Real, b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUComponentVecorMat},
                            B::AbstractGPUComponentVecorMat, a::Real, b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat},
                            B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Real,
                            b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUComponentVecorMat},
                            B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUComponentVecorMat}, a::Real,
                            b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat},
                            B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat},
                            a::Real, b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUComponentVecorMat},
                            B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUComponentVecorMat},
                            a::Real, b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat},
                            B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUVecOrMat}, a::Real,
                            b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUComponentVecorMat},
                            B::LinearAlgebra.Adjoint{<:Any, <:AbstractGPUComponentVecorMat}, a::Real,
                            b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat},
                            B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUVecOrMat},
                            a::Real, b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
function LinearAlgebra.mul!(C::AbstractGPUComponentVecorMat,
                            A::LinearAlgebra.Transpose{<:Any, <:AbstractGPUComponentVecorMat},
                            B::LinearAlgebra.Transpose{<:Any, <:AbstractGPUComponentVecorMat},
                            a::Real, b::Real)
    return GPUArrays.generic_matmatmul!(C, A, B, a, b)
end
