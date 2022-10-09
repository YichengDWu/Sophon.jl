abstract type PINNSampler end

"""
    sample(pde::PDESystem, sampler::PINNSampler, strategy=nothing)

Sample the datasets for the PDEs and boundary conditions using the given sampler.
"""
function sample end

"""
    QuasiRandomSampler(pde_points, bcs_points=pde_points;
                       sampling_alg=SobolSample())

Sampler to generate the datasets for PDE and boundary conditions using a quisa-random sampling algorithm.
You can call `sample(pde, sampler, strategy)` on it to generate all the datasets. See [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl)
for available sampling algorithms. The default element type of the sampled data is `Float64`. The initial
sampled data lives on GPU if [`PINN`](@ref) is. You will need manually move the data to GPU if you want to resample.
"""
struct QuasiRandomSampler{P, B, S} <: PINNSampler
    pde_points::P
    bcs_points::B
    sampling_alg::S
end

function QuasiRandomSampler(pde_points, bcs_points=pde_points; sampling_alg=SobolSample())
    return QuasiRandomSampler{typeof(pde_points), typeof(bcs_points), typeof(sampling_alg)}(pde_points,
                                                                                            bcs_points,
                                                                                            sampling_alg)
end

function sample(pde::NeuralPDE.PDESystem, sampler::QuasiRandomSampler, strategy)
    (; pde_points, bcs_points, sampling_alg) = sampler
    pde_bounds, bcs_bounds = get_bounds(pde)

    pde_points = length(pde_points) == 1 ?
                 ntuple(_ -> first(pde_points), length(pde_bounds)) : Tuple(pde_points)
    bcs_points = length(bcs_points) == 1 ?
                 ntuple(_ -> first(bcs_points), length(bcs_bounds)) : Tuple(bcs_points)

    pde_datasets = [QuasiMonteCarlo.sample(points, bound[1], bound[2], sampling_alg)
                    for (points, bound) in zip(pde_points, pde_bounds)]

    boundary_datasets = [QuasiMonteCarlo.sample(points, bound[1], bound[2],
                                                sampling_alg)
                         for (points, bound) in zip(bcs_points, bcs_bounds)]

    return [pde_datasets; boundary_datasets]
end

function sample(pde::PDESystem, sampler::QuasiRandomSampler, strategy)
    (; pde_points, bcs_points, sampling_alg) = sampler
    (; eqs, bcs) = pde

    pde_points = length(pde_points) == 1 ?
                 ntuple(_ -> first(pde_points), length(eqs)) : Tuple(pde_points)
    bcs_points = length(bcs_points) == 1 ?
                 ntuple(_ -> first(bcs_points), length(bcs)) : Tuple(bcs_points)

    pde_datasets = [sample(eq[2], points, sampling_alg) for (eq, points) in zip(eqs, pde_points)]
    boundary_datasets = [sample(bcs[2], points, sampling_alg) for (bcs, points) in zip(bcs, bcs_points)]

    return [pde_datasets; boundary_datasets]
end


function sample(pde::NeuralPDE.PDESystem, sampler::QuasiRandomSampler{P, B, SobolSample},
                strategy) where {P, B}
    (; pde_points, bcs_points) = sampler
    pde_bounds, bcs_bounds = get_bounds(pde)

    pde_points = length(pde_points) == 1 ?
                 ntuple(_ -> first(pde_points), length(pde_bounds)) : Tuple(pde_points)

    bcs_points = length(bcs_points) == 1 ?
                 ntuple(_ -> first(bcs_points), length(bcs_bounds)) : Tuple(bcs_points)

    pde_datasets = [sobolsample(points, bound[1], bound[2])
                    for (points, bound) in zip(pde_points, pde_bounds)]

    boundary_datasets = [sobolsample(points, bound[1], bound[2])
                         for (points, bound) in zip(bcs_points, bcs_bounds)]

    return [pde_datasets; boundary_datasets]
end

function sample(d::Rectangle, points::Int, ::SobolSample)
    bounds = get_bounds(d)
    return sobolsample(points, bounds[1], bounds[2])
end

function sample(d::Interval, points::Int, ::SobolSample)
    bounds = get_bounds(d)
    return sobolsample(points, bounds[1], bounds[2])
end

function sample(d::SetdiffDomain{S, <:Tuple{<:Rectangle, F}}, points::Int, alg::QuasiMonteCarlo.SamplingAlgorithm) where {S, F}
    rec = d.domains[1]
    data = sample(rec, points, alg)
    idx = [x ∈ d for d in eachcol(data)]
    return data[:, idx]
end

function sample(d::Rectangle, points::Int, sampling_alg::QuasiMonteCarlo.SamplingAlgorithm)
    bounds = get_bounds(d)
    return QuasiMonteCarlo.sample(points, bounds[1], bounds[2], sampling_alg)
end

function sample(d::Interval, points::Int, sampling_alg::QuasiMonteCarlo.SamplingAlgorithm)
    bounds = get_bounds(d)
    return QuasiMonteCarlo.sample(points, bounds[1], bounds[2], sampling_alg)
end

function sobolsample(n::Int, lb, ub)
    s = cached_sobolseq(n, lb, ub)
    return reduce(hcat, [Sobol.next!(s) for i in 1:n])
end

@memoize LRU{Tuple{Int, AbstractVector, AbstractVector}, Any}(maxsize=100) function cached_sobolseq(n, lb,
                                                                                    ub)
    s = Sobol.SobolSeq(lb, ub)
    s = Sobol.skip(s, n)
    return s
end
