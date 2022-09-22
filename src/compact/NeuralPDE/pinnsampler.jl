abstract type PINNSampler{T, N1, N2} end

"""
    sample(pde::PDESystem, sampler::PINNSampler, strategy=nothing)

Sample the datasets for the PDEs and boundary conditions using the given sampler.
"""
function sample(::PDESystem, ::PINNSampler, ::AbstractTrainingAlg) end

"""
    QuasiRandomSampler(pde_points, bcs_points=pde_points;
                       device_type::Type=Array{Float64}
                       sampling_alg=LatinHypercubeSample())

Sampler to generate the datasets for PDE and boundary conditions using a quisa-random sampling algorithm.
It momerizes the domain of the PDE and the boundary conditions, and you can call `sample` on it to generate the datasets.
"""
struct QuasiRandomSampler{T, N1, N2, S} <: PINNSampler{T, N1, N2}
    pde_points::NTuple{N1, Int}
    bcs_points::NTuple{N2, Int}
    sampling_alg::S
end

function QuasiRandomSampler(pde_points, bcs_points=pde_points;
                            device_type::Type=Array{Float64},
                            sampling_alg=LatinHypercubeSample())
    N1 = length(pde_bounds)
    N2 = length(bcs_bounds)
    pde_points = length(pde_points) == 1 ? ntuple(_ -> first(pde_points), N1) :
                 Tuple(pde_points)
    bcs_points = length(bcs_points) == 1 ? ntuple(_ -> first(bcs_points), N2) :
                 Tuple(bcs_points)

    return QuasiRandomSampler{device_type, N1, N2, typeof(sampling_alg)}(pde_points,
                                                                         bcs_points,
                                                                         sampling_alg)
end

function sample(pde::PDESystem, sampler::QuasiRandomSampler{device_type}, strategy) where {device_type}
    eltypeθ = eltype(device_type)
    (; pde_points, bcs_points, sampling_alg) = sampler
    pde_bounds, bcs_bounds = get_bounds(pde)

    pde_datasets = [NeuralPDE.generate_quasi_random_points(points, bound, eltypeθ,
                                                           sampling_alg)
                    for (points, bound) in zip(pde_points, pde_bounds)]
    pde_datasets = [adapt(device_type, pde_dataset) for pde_dataset in pde_datasets]

    boundary_datasets = [NeuralPDE.generate_quasi_random_points(points, bound, eltypeθ,
                                                                sampling_alg)
                         for (points, bound) in zip(bcs_points, bcs_bounds)]

    boundary_datasets = [adapt(device_type, boundary_dataset)
                         for boundary_dataset in boundary_datasets]

    return pde_datasets, boundary_datasets
end
