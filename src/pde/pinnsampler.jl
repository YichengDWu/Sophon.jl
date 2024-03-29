abstract type PINNSampler end
abstract type FunctionSampler end

"""
    sample(pde::PDESystem, sampler::PINNSampler, strategy=nothing)

Sample the datasets for the PDEs and boundary conditions using the given sampler.
"""
function sample end

"""
    QuasiRandomSampler(pde_points, bcs_points=pde_points;
                       sampling_alg=SobolSample(),
                       resample = false))

Sampler to generate the datasets for PDE and boundary conditions using a quisa-random sampling algorithm.
You can call `sample(pde, sampler, strategy)` on it to generate all the datasets. See [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl)
for available sampling algorithms. The default element type of the sampled data is `Float64`. The initial
sampled data lives on GPU if [`PINN`](@ref) is. You will need manually move the data to GPU if you want to resample.

## Arguments

  - `pde_points`: The number of points to sample for each PDE. If a single number is given, the same number of points
    will be sampled for each PDE. If a tuple of numbers is given, the number of points for each PDE will be the
    corresponding element in the tuple. The default is `100`.
  - `bcs_points`: The number of points to sample for each boundary condition. If a single number is given, the same
    number of points will be sampled for each boundary condition. If a tuple of numbers is given, the number of
    points for each boundary condition will be the corresponding element in the tuple. The default is `pde_points`.

## Keyword Arguments

  - `sampling_alg`: The sampling algorithm to use. The default is `SobolSample()`.

  - `resample`: Whether to resample the data for each equation. The default is `false`, which can save a lot of memory
    if you are solving a large number of PDEs. In this case, `pde_points` has to be a integer. If you want to resample the data, you will need to manually move
    the data to GPU if you want to use GPU to solve the PDEs.
"""
struct QuasiRandomSampler{R, P, B, S} <: PINNSampler
    pde_points::P
    bcs_points::B
    sampling_alg::S
end

function QuasiRandomSampler(pde_points, bcs_points=pde_points; sampling_alg=SobolSample(),
                            resample::Bool=false)
    return QuasiRandomSampler{resample, typeof(pde_points), typeof(bcs_points),
                              typeof(sampling_alg)}(pde_points, bcs_points, sampling_alg)
end

function sample(pde::ModelingToolkit.PDESystem, sampler::QuasiRandomSampler{true})
    (; pde_points, bcs_points, sampling_alg) = sampler
    pde_bounds, bcs_bounds = get_bounds(pde)

    pde_points = length(pde_points) == 1 ?
                 ntuple(_ -> first(pde_points), length(pde_bounds)) : Tuple(pde_points)
    bcs_points = length(bcs_points) == 1 ?
                 ntuple(_ -> first(bcs_points), length(bcs_bounds)) : Tuple(bcs_points)

    pde_datasets = [QuasiMonteCarlo.sample(points, bound[1], bound[2], sampling_alg)
                    for (points, bound) in zip(pde_points, pde_bounds)]

    boundary_datasets = [QuasiMonteCarlo.sample(points, bound[1], bound[2], sampling_alg)
                         for (points, bound) in zip(bcs_points, bcs_bounds)]

    return [pde_datasets; boundary_datasets]
end

function sample(pde::ModelingToolkit.PDESystem, sampler::QuasiRandomSampler{false, <:Int})
    (; pde_points, bcs_points, sampling_alg) = sampler
    pde_bounds, bcs_bounds = get_bounds(pde)

    bcs_points = length(bcs_points) == 1 ?
                 ntuple(_ -> first(bcs_points), length(bcs_bounds)) : Tuple(bcs_points)

    pde_dataset = QuasiMonteCarlo.sample(pde_points, pde_bounds[1][1], pde_bounds[1][2],
                                         sampling_alg)
    pde_datasets = [pde_dataset for _ in 1:length(pde_bounds)]

    boundary_datasets = [QuasiMonteCarlo.sample(points, bound[1], bound[2], sampling_alg)
                         for (points, bound) in zip(bcs_points, bcs_bounds)]

    return [pde_datasets; boundary_datasets]
end

function sample(pde::Sophon.PDESystem, sampler::QuasiRandomSampler)
    (; pde_points, bcs_points, sampling_alg) = sampler
    (; eqs, bcs) = pde

    pde_points = length(pde_points) == 1 ? ntuple(_ -> first(pde_points), length(eqs)) :
                 Tuple(pde_points)
    bcs_points = length(bcs_points) == 1 ? ntuple(_ -> first(bcs_points), length(bcs)) :
                 Tuple(bcs_points)

    pde_datasets = [sample(eq[2], points, sampling_alg)
                    for (eq, points) in zip(eqs, pde_points)]
    boundary_datasets = [sample(bc[2], points, sampling_alg)
                         for (bc, points) in zip(bcs, bcs_points)]

    return [pde_datasets; boundary_datasets]
end

function sample(pde::ModelingToolkit.PDESystem,
                sampler::QuasiRandomSampler{true, P, B, SobolSample}) where {P, B}
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

function sample(pde::ModelingToolkit.PDESystem,
                sampler::QuasiRandomSampler{false, P, B, SobolSample}) where {P<:Int, B}
    (; pde_points, bcs_points) = sampler
    pde_bounds, bcs_bounds = get_bounds(pde)

    bcs_points = length(bcs_points) == 1 ?
                 ntuple(_ -> first(bcs_points), length(bcs_bounds)) : Tuple(bcs_points)

    pde_dataset = sobolsample(pde_points, pde_bounds[1][1], pde_bounds[1][2])
    pde_datasets = [pde_dataset for _ in 1:length(pde_bounds)]

    boundary_datasets = [sobolsample(points, bound[1], bound[2])
                         for (points, bound) in zip(bcs_points, bcs_bounds)]

    return [pde_datasets; boundary_datasets]
end

"""
    BetaRandomSampler(pde_points, bcs_points=pde_points; sampling_alg=SobolSample(),
                      resample::Bool=false, α=0.4, β=1.0)
Same as `QuasiRandomSampler`, but use `Beta` distribution along time on the domain.
"""
struct BetaRandomSampler{R,P,B,S,A,L} <: PINNSampler
    pde_points::P
    bcs_points::B
    sampling_alg::S
    α::A
    β::L
end

function BetaRandomSampler(pde_points, bcs_points=pde_points; sampling_alg=SobolSample(),
                            resample::Bool=false, α=0.4, β=1.0)
    return BetaRandomSampler{resample, typeof(pde_points), typeof(bcs_points),
                              typeof(sampling_alg), typeof(α), typeof(β)}(pde_points, bcs_points,
                                                                          sampling_alg, α, β)
end

function SciMLBase.remake(sampler::BetaRandomSampler{resample}; α) where {resample}
    (; pde_points, bcs_points, sampling_alg, β) = sampler
    return BetaRandomSampler{resample, typeof(pde_points), typeof(bcs_points),
                              typeof(sampling_alg), typeof(α), typeof(β)}(pde_points, bcs_points,
                                                                          sampling_alg, α, β)
end

function sample(pde::ModelingToolkit.PDESystem, sampler::BetaRandomSampler{false, <:Int})
    t_pos = get_where_t_is(pde)
    (; pde_points, bcs_points, sampling_alg, α, β) = sampler
    pde_bounds, bcs_bounds = get_bounds(pde)

    tspan = pde.domain[t_pos].domain.left, pde.domain[t_pos].domain.right
    pde_bounds = [(deleteat!(lb,t_pos),  deleteat!(ub,t_pos)) for (lb, ub) in pde_bounds]

    bcs_points = length(bcs_points) == 1 ?
                 ntuple(_ -> first(bcs_points), length(bcs_bounds)) : Tuple(bcs_points)

    pde_dataset = if isempty(pde_bounds[1][1])
        zeros(0, pde_points)
    else
        QuasiMonteCarlo.sample(pde_points, pde_bounds[1][1], pde_bounds[1][2],
                                            sampling_alg)
    end

    ts = rand(Beta(α, β), (1, pde_points))
    ts = tspan[1] .+ (tspan[2] .- tspan[1]) .* ts
    pde_dataset = insert_row(pde_dataset, ts, t_pos)
    pde_datasets = [pde_dataset for _ in 1:length(pde_bounds)]

    boundary_datasets = [QuasiMonteCarlo.sample(points, bound[1], bound[2], sampling_alg)
                         for (points, bound) in zip(bcs_points, bcs_bounds)]

    return [pde_datasets; boundary_datasets]
end

function sample(d::DomainSets.GenericBall{Vector{T}}, points::Int,
                alg::QuasiMonteCarlo.SamplingAlgorithm) where {T}
    (; center, radius) = d
    return sample(Ball(radius, StaticArraysCore.SVector(center...)), points, alg)
end

function sample(d::DomainSets.Disk, points::Int, alg::QuasiMonteCarlo.SamplingAlgorithm)
    (; center, radius) = d
    xys = sample(points, [-1.0, -1.0], [1.0, 1.0], alg)
    data = [ifelse(abs(xy[1]) ≥ abs(xy[2]),
                   [xy[1] * cos(π / 4 * xy[2] / xy[1]), xy[1] * sin(π / 4 * xy[2] / xy[1])],
                   [xy[2] * sin(π / 4 * xy[1] / xy[2]), xy[2] * cos(π / 4 * xy[1] / xy[2])])
            for xy in eachcol(xys)]
    data = reduce(hcat, data)
    data = center .+ radius .* data
    return data
end

function sample(d::DomainSets.GenericSphere{Vector{T}, T}, points::Int,
                alg::QuasiMonteCarlo.SamplingAlgorithm) where {T}
    (; center, radius) = d
    return sample(Sphere(radius, StaticArraysCore.SVector(center...)), points, alg)
end

function sample(d::DomainSets.GenericSphere{StaticArraysCore.SVector{2, T}, T}, points::Int,
                alg::QuasiMonteCarlo.SamplingAlgorithm) where {T}
    (; center, radius) = d
    θ = sample(points, [0.0], [2π], alg)
    data = center .+ radius .* [cos.(θ); sin.(θ)]
    return data
end

function sample(d::DomainSets.UnitCircle, points::Int,
                alg::QuasiMonteCarlo.SamplingAlgorithm)
    θ = sample(points, [0.0], [2π], alg)
    data = [cos.(θ); sin.(θ)]
    return data
end

function sample(d::DomainSets.GenericSphere{StaticArraysCore.SVector{3, T}, T}, points::Int,
                alg::QuasiMonteCarlo.SamplingAlgorithm) where {T}
    (; center, radius) = d
    r = sample(points, [-1, -1], [1, 1], alg)
    r = r[:, [x[1]^2 + x[2]^2 <= 1 for x in eachcol(r)]]
    x1 = r[1:1, :]
    x2 = r[2:2, :]
    x = @. 2 * x1 * sqrt(1 - x1^2 - x2^2)
    y = @. 2 * x2 * sqrt(1 - x1^2 - x2^2)
    z = @. 1 - 2 * (x1^2 + x2^2)
    data = [x; y; z]
    data = center .+ radius .* data
    return data
end

function sample(d::SetdiffDomain{S, <:Tuple{<:Rectangle, F}}, points::Int,
                alg::QuasiMonteCarlo.SamplingAlgorithm) where {S, F}
    rec = d.domains[1]
    data = sample(rec, points, alg)
    idx = [x ∈ d for x in eachcol(data)]
    return data[:, idx]
end

function sample(d::UnionDomain, points::Int, alg::QuasiMonteCarlo.SamplingAlgorithm)
    data = mapreduce(x -> sample(x, points, alg), hcat, d.domains)
    return data
end

function sample(d::Rectangle, points::Int, sampling_alg::QuasiMonteCarlo.SamplingAlgorithm)
    bounds = get_bounds(d)
    return sample(points, bounds[1], bounds[2], sampling_alg)
end

function sample(d::Interval, points::Int, sampling_alg::QuasiMonteCarlo.SamplingAlgorithm)
    bounds = get_bounds(d)
    return sample(points, bounds[1], bounds[2], sampling_alg)
end

function sample(points::Int, lb::AbstractVector, ub::AbstractVector,
                sampling_alg::QuasiMonteCarlo.SamplingAlgorithm)
    return QuasiMonteCarlo.sample(points, lb, ub, sampling_alg)
end

function sample(points::Int, lb::AbstractVector, ub::AbstractVector, ::SobolSample)
    return sobolsample(points, lb, ub)
end

function sobolsample(n::Int, lb, ub)
    s = cached_sobolseq(n, lb, ub)
    return reduce(hcat, [Sobol.next!(s) for i in 1:n])
end

@memoize LRU{Tuple{Int, AbstractVector, AbstractVector}, Any}(maxsize=100) function cached_sobolseq(n,
                                                                                                    lb,
                                                                                                    ub)
    s = Sobol.SobolSeq(lb, ub)
    s = Sobol.skip(s, n)
    return s
end
