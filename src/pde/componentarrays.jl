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

function _ComponentArray(nt::NamedTuple)
    return isongpu(nt) ? adapt(get_gpu_adaptor(), ComponentArray(cpu(nt))) : ComponentArray(nt)
end
