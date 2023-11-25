module SophonTaylorDiffLuxCUDAExt

using TaylorDiff, LuxCUDA, Sophon

function Base.:*(A::Union{Sophon.CuMatrix{T}, LinearAlgebra.Transpose{T, Sophon.CuArray}},
                 B::Sophon.CuMatrix{TaylorScalar{T, N}}) where {T, N}
    C = similar(B, (size(A, 1), size(B, 2)))
    fill!(C, zero(eltype(C)))
    return LinearAlgebra.mul!(C, A, B)
end

end
