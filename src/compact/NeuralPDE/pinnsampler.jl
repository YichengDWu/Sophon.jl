abstract type PINNSampler{T, N1, N2} end

"""
    QuasiRandomSampler(pde::NeuralPDE.PDESystem, pde_points, bcs_points=pde_points;
                       device_type::Type=Array{Float64}
                       sampling_alg=LatinHypercubeSample())

Sampler to generate the datasets for PDE and boundary conditions using a quisa-random sampling algorithm.
It momerizes the domain of the PDE and the boundary conditions, and you can call `sample` on it to generate the datasets.
"""
struct QuasiRandomSampler{T, N1, N2, S} <: PINNSampler{T, N1, N2}
    pde_points::NTuple{N1, Int}
    bcs_points::NTuple{N2, Int}
    pde_bounds::Any
    bcs_bounds::Any
    sampling_alg::S
end

function QuasiRandomSampler(pde::NeuralPDE.PDESystem, pde_points, bcs_points=pde_points;
                            device_type::Type=Array{Float64},
                            sampling_alg=LatinHypercubeSample())
    (; eqs, bcs, domain, ivs, dvs) = pde
    _, _, dict_indvars, dict_depvars, _ = NeuralPDE.get_vars(ivs, dvs)
    bounds = get_bounds(domain, eqs, bcs, eltype(device_type), dict_indvars, dict_depvars)
    pde_bounds, bcs_bounds = bounds

    N1 = length(pde_bounds)
    N2 = length(bcs_bounds)
    pde_points = length(pde_points) == 1 ? ntuple(_ -> first(pde_points), N1) :
                 Tuple(pde_points)
    bcs_points = length(bcs_points) == 1 ? ntuple(_ -> first(bcs_points), N2) :
                 Tuple(bcs_points)

    return QuasiRandomSampler{device_type, N1, N2, typeof(sampling_alg)}(pde_points,
                                                                         bcs_points,
                                                                         pde_bounds,
                                                                         bcs_bounds,
                                                                         sampling_alg)
end

function sample(pinnsampler::QuasiRandomSampler{device_type}) where {device_type}
    eltypeθ = eltype(device_type)
    (; pde_points, bcs_points, pde_bounds, bcs_bounds, sampling_alg) = pinnsampler

    # generate_quasi_random_point does not respect data type
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