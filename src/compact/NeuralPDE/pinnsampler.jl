abstract type PINNSampler{T} end

"""
    sample(pde::PDESystem, sampler::PINNSampler, strategy=nothing)

Sample the datasets for the PDEs and boundary conditions using the given sampler.
"""
function sample end

"""
    QuasiRandomSampler(pde_points, bcs_points=pde_points;
                       device_type::Type=Array{Float64}
                       sampling_alg=SobolSample())

Sampler to generate the datasets for PDE and boundary conditions using a quisa-random sampling algorithm.
It momerizes the domain of the PDE and the boundary conditions, and you can call `sample` on it to generate the datasets.
"""
struct QuasiRandomSampler{T, P, B, S} <: PINNSampler{T}
    pde_points::P
    bcs_points::B
    sampling_alg::S
end

function QuasiRandomSampler(pde_points, bcs_points=pde_points;
                            device_type::Type=Array{Float64},
                            sampling_alg=SobolSample())
    return QuasiRandomSampler{device_type, typeof(pde_points), typeof(bcs_points),
                              typeof(sampling_alg)}(pde_points, bcs_points, sampling_alg)
end

function sample(pde::PDESystem, sampler::QuasiRandomSampler{device_type},
                strategy) where {device_type}
    eltypeθ = eltype(device_type)
    (; pde_points, bcs_points, sampling_alg) = sampler
    pde_bounds, bcs_bounds = get_bounds(pde)

    pde_points = length(pde_points) == 1 ?
                 ntuple(_ -> first(pde_points), length(pde_bounds)) : Tuple(pde_points)
    bcs_points = length(bcs_points) == 1 ?
                 ntuple(_ -> first(bcs_points), length(bcs_bounds)) : Tuple(bcs_points)

    pde_datasets = [QuasiMonteCarlo.sample(points, bound[1], bound[2], sampling_alg)
                    for (points, bound) in zip(pde_points, pde_bounds)]
    pde_datasets = [adapt(device_type, pde_dataset) for pde_dataset in pde_datasets]

    boundary_datasets = [QuasiMonteCarlo.sample(points, bound[1], bound[2], sampling_alg)
                         for (points, bound) in zip(bcs_points, bcs_bounds)]

    boundary_datasets = [adapt(device_type, boundary_dataset)
                         for boundary_dataset in boundary_datasets]

    return [pde_datasets; boundary_datasets]
end

function sample(pde::PDESystem, sampler::QuasiRandomSampler{device_type, P, B, SobolSample},
                strategy) where {device_type, P, B}
    eltypeθ = eltype(device_type)
    (; pde_points, bcs_points) = sampler
    pde_bounds, bcs_bounds = get_bounds(pde)

    pde_points = length(pde_points) == 1 ?
                 ntuple(_ -> first(pde_points), length(pde_bounds)) : Tuple(pde_points)
    bcs_points = length(bcs_points) == 1 ?
                 ntuple(_ -> first(bcs_points), length(bcs_bounds)) : Tuple(bcs_points)

    pde_datasets = [sobolsample(points, bound[1], bound[2])
                    for (points, bound) in zip(pde_points, pde_bounds)]
    pde_datasets = [adapt(device_type, pde_dataset) for pde_dataset in pde_datasets]

    boundary_datasets = [sobolsample(points, bound[1], bound[2])
                         for (points, bound) in zip(bcs_points, bcs_bounds)]

    boundary_datasets = [adapt(device_type, boundary_dataset)
                         for boundary_dataset in boundary_datasets]

    return [pde_datasets; boundary_datasets]
end

function sobolsample(n::Int, lb, ub)
    s = cached_sobolseq(lb, ub)
    return reduce(hcat, [Sobol.next!(s) for i in 1:n])
end

@memoize LRU{Tuple{Vector, Vector}, Any}(maxsize=100) function cached_sobolseq(lb,ub)
    s = Sobol.SobolSeq(lb, ub)
    s = Sobol.skip(s, 1000)
    return s
end
