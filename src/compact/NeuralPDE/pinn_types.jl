@inline NeuralPDE.Phi(phi::NeuralPDE.Phi) = phi

function NeuralPDE.Phi(f, st::NamedTuple)
    return NeuralPDE.Phi{typeof(f), typeof(st)}(f, st)
end

function initialparameters(rng::AbstractRNG, phi::NeuralPDE.Phi{<:Lux.AbstractExplicitLayer})
    return initialparameters(rng, phi.f)
end

Lux.cpu(phi::NeuralPDE.Phi{<:Lux.AbstractExplicitLayer}) = NeuralPDE.Phi(phi.f, cpu(phi.st))


Lux.gpu(phi::NeuralPDE.Phi{<:Lux.AbstractExplicitLayer}) = NeuralPDE.Phi(phi.f, gpu(phi.st))
