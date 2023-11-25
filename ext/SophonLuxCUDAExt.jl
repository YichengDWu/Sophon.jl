module SophonLuxCUDAExt

using Lux, LuxCUDA, Sophon, Optimization, Adapt

function (::LuxCUDADevice)(prob::OptimizationProblem)
    u0 = adapt(CuArray, prob.u0)
    p = [adapt(CuArray, prob.p[i]) for i in 1:length(prob.p)]
    prob = remake(prob, u0=u0, p=p)
    return prob
end

end
