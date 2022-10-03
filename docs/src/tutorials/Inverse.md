
# Lorenz System

$\frac{\mathrm{d} x}{\mathrm{~d} t}=\sigma(y-x)$,

$\frac{\mathrm{d} y}{\mathrm{~d} t}=x(\rho-z)-y$,

$\frac{\mathrm{d} z}{\mathrm{~d} t}=x y-\beta z$,


```julia
using ModelingToolkit, Sophon,Lux,OrdinaryDiffEq
using Optimization, OptimizationOptimJL
using ModelingToolkit, IntervalSets
```


```julia
@parameters t 
@variables x(..), y(..), z(..),σ_(..) ,β(..), ρ(..)
Dt = Differential(t)
eqs = [Dt(x(t)) ~ σ_(t)*(y(t) - x(t)),
       Dt(y(t)) ~ x(t)*(ρ(t) - z(t)) - y(t),
       Dt(z(t)) ~ x(t)*y(t) - β(t)*z(t)]

bcs = [x(0) ~ 1.0, y(0) ~ 0.0, z(0) ~ 0.0]
domains = [t ∈ Interval(0.0,1.0)]
@named pde_system = PDESystem(eqs, bcs, domains, [t], [x(t),y(t),z(t),σ_(t), ρ(t), β(t)])
```




$$ \begin{align}
\frac{dx(t)}{dt} =& \left(  - x\left( t \right) + y\left( t \right) \right) \sigma_{}\left( t \right) \\
\frac{dy(t)}{dt} =&  - y\left( t \right) + \left(  - z\left( t \right) + \rho\left( t \right) \right) x\left( t \right) \\
\frac{dz(t)}{dt} =& x\left( t \right) y\left( t \right) - z\left( t \right) \beta\left( t \right)
\end{align}
 $$




```julia
function lorenz!(du,u,p,t)
 du[1] = 10.0*(u[2]-u[1])
 du[2] = u[1]*(28.0-u[3]) - u[2]
 du[3] = u[1]*u[2] - (8/3)*u[3]
end

u0 = [1.0;0.0;0.0]
tspan = (0.0,1.0)
prob = ODEProblem(lorenz!,u0,tspan)
sol = solve(prob, Tsit5(), dt=0.1)
ts = [infimum(d.domain):0.1:supremum(d.domain) for d in domains][1]
function getData(sol)
    data = []
    us = hcat(sol(ts).u...)
    ts_ = hcat(sol(ts).t...)
    return [us,ts_]
end
data = getData(sol)

(u_ , t_) = data
```




    2-element Vector{Matrix{Float64}}:
     [1.0 1.2714705245515625 … -9.129935274368785 -9.397841485892151; 0.0 2.5502284813556653 … -9.919199673331995 -9.094622603195736; 0.0 0.11415728662964351 … 26.980024057777484 28.558082735246224]
     [0.0 0.1 … 0.9 1.0]




```julia
input_=1
n=2
pinn = PINN(x  =Lux.Chain(Dense(input_, n, Lux.sech), Dense(n, n, Lux.sech),Dense(n, n, Lux.sech), Dense(n, 1)),
            y  =Lux.Chain(Dense(input_, n, Lux.sech), Dense(n, n, Lux.sech),Dense(n, n, Lux.sech), Dense(n, 1)),
            z  =Lux.Chain(Dense(input_, n, Lux.sech), Dense(n, n, Lux.sech),Dense(n, n, Lux.sech), Dense(n, 1)),
            σ_=ConstantFunction(),
            ρ =ConstantFunction(),
            β =ConstantFunction())
sampler = QuasiRandomSampler(2000, 500)
strategy = NonAdaptiveTraining(1,0)

t_data = t_
u_data = u_ 
function additional_loss(phi, θ)
    return sum(abs2, vcat(phi.x(t_data, θ.x), phi.y(t_data, θ.y), phi.z(t_data, θ.z)).-u_data)/length(t_data)
end
prob = Sophon.discretize(pde_system, pinn, sampler, strategy, additional_loss=additional_loss)

callback = function (p,l)
    println("Current loss is: $l")
    return false
end
@time res = Optimization.solve(prob, BFGS(), callback = callback, maxiters=1000)
```

    Current loss is: 1.602728259296474
    Current loss is: 0.43765052039740554
    Current loss is: 0.1301403270380814
    Current loss is: 0.02182199542085611
    Current loss is: 0.008052007116189768
    Current loss is: 0.0008446550476317993
    Current loss is: 0.00023949212817193157
    Current loss is: 4.816915708560392e-5
    Current loss is: 2.244131431052732e-5
    Current loss is: 2.1668910570636866e-5
    Current loss is: 1.1427742612493897e-5
    Current loss is: 1.097623583987271e-5
    Current loss is: 1.2065182611567046e-6
    Current loss is: 7.847955663308717e-7
    Current loss is: 7.634347229970087e-7
    Current loss is: 7.606956388588401e-7
    Current loss is: 7.591493692951559e-7
    Current loss is: 6.432107794150076e-7
    Current loss is: 6.369426607762923e-7
    Current loss is: 6.347132825323099e-7
    Current loss is: 3.779061188155574e-7
    Current loss is: 3.0428453443336387e-7
    Current loss is: 2.8197997394568e-7
    Current loss is: 2.7669069385869357e-7
    Current loss is: 2.7599524550972e-7
    Current loss is: 2.7578917969193105e-7
    Current loss is: 2.6488231835247944e-7
    Current loss is: 8.070419951524142e-8
    Current loss is: 4.291680530327459e-8
    Current loss is: 2.7406831800262062e-8
    Current loss is: 2.030748620395981e-8
    Current loss is: 1.5426380096524868e-8
    Current loss is: 1.0784211446398811e-8
    Current loss is: 6.837126749432508e-9
    Current loss is: 4.931757319363791e-9
    Current loss is: 4.519526660701046e-9
    Current loss is: 3.5934941170876744e-9
    Current loss is: 3.2034307843991195e-9
    Current loss is: 2.7679293604166707e-9
    Current loss is: 2.4428377291670554e-9
    Current loss is: 2.3204569303062668e-9
    Current loss is: 2.261395169890597e-9
    Current loss is: 2.181361776108589e-9
    Current loss is: 2.1590712605188498e-9
    Current loss is: 2.1511938244114547e-9
    Current loss is: 2.1424756482911488e-9
    Current loss is: 2.1330655396432584e-9
    Current loss is: 2.1118577961646186e-9
    Current loss is: 2.0703719594682285e-9
    Current loss is: 2.0632452339284924e-9
    Current loss is: 2.0624839777615496e-9
    Current loss is: 2.0560169147262827e-9
    Current loss is: 2.0516655830279893e-9
    Current loss is: 2.0453655804622998e-9
    Current loss is: 2.043363518854558e-9
    Current loss is: 2.0407161385663427e-9
    Current loss is: 2.039201425443485e-9
    Current loss is: 2.0375953148703808e-9
    Current loss is: 2.0364484484110752e-9
    Current loss is: 2.036030476422257e-9
    Current loss is: 2.035523438639577e-9
    Current loss is: 2.0353583759752085e-9
    Current loss is: 2.0350840976063337e-9
    Current loss is: 2.0349674117105072e-9
    Current loss is: 2.034936189383765e-9
    Current loss is: 2.034883911476809e-9
    Current loss is: 2.034812236896001e-9
    Current loss is: 2.0347211229709067e-9
    Current loss is: 2.034566354357358e-9
    Current loss is: 2.0341283731634988e-9
    Current loss is: 2.0337831169789633e-9
    Current loss is: 2.033533352552261e-9
    Current loss is: 2.03304315450403e-9
    Current loss is: 2.0319140721477885e-9
    Current loss is: 2.0292259919038965e-9
    Current loss is: 2.026911706897422e-9
    Current loss is: 2.014934466344822e-9
    Current loss is: 2.010458549065763e-9
    Current loss is: 2.0019619667800645e-9
    Current loss is: 1.9887302547740875e-9
    Current loss is: 1.966284930549313e-9
    Current loss is: 1.9221653144532565e-9
    Current loss is: 1.8392261457373639e-9
    Current loss is: 1.8346500336310276e-9
    Current loss is: 1.7898284529984498e-9
    Current loss is: 1.7477411635085972e-9
    Current loss is: 1.7146625791255126e-9
    Current loss is: 1.7034889302656382e-9
    Current loss is: 1.6873483215398051e-9
    Current loss is: 1.6828428003373156e-9
    Current loss is: 1.6819635404199558e-9
    Current loss is: 1.6816617955329287e-9
    Current loss is: 1.6815970539988843e-9
    Current loss is: 1.6815609278493227e-9
    Current loss is: 1.677296419563252e-9
    Current loss is: 1.673045011052997e-9
    Current loss is: 1.6658031630955975e-9
    Current loss is: 1.6607987295176046e-9
    Current loss is: 1.6552701220845451e-9
    Current loss is: 1.6500309839670664e-9
    Current loss is: 1.6474241084226793e-9
    Current loss is: 1.641056477560507e-9
    Current loss is: 1.6381401802043941e-9
    Current loss is: 1.6349420777035497e-9
    Current loss is: 1.6307517538627642e-9
    Current loss is: 1.627277587795194e-9
    Current loss is: 1.6247016669398757e-9
    Current loss is: 1.6231229857933452e-9
    Current loss is: 1.621503174302344e-9
    Current loss is: 1.6195904845717367e-9
    Current loss is: 1.6181210957336162e-9
    Current loss is: 1.6156309006853464e-9
    Current loss is: 1.61409796582199e-9
    Current loss is: 1.611773035552275e-9
    Current loss is: 1.6108582721177477e-9
    Current loss is: 1.6089259974938838e-9
    Current loss is: 1.6081023310587615e-9
    Current loss is: 1.6075047316791755e-9
    Current loss is: 1.6070185649382495e-9
    Current loss is: 1.6064568456493622e-9
    Current loss is: 1.6060477887052875e-9
    Current loss is: 1.6056632525011737e-9
    Current loss is: 1.6055062173114222e-9
    Current loss is: 1.6053860940898838e-9
    Current loss is: 1.6052403562753323e-9
    Current loss is: 1.6052228591293325e-9
    Current loss is: 1.6052184359637197e-9
    Current loss is: 1.6052053577311649e-9
    Current loss is: 1.6051465119328797e-9
    Current loss is: 1.6049717974477715e-9
    Current loss is: 1.6046786847409108e-9
    Current loss is: 1.6033789455006873e-9
    Current loss is: 1.6017586909255686e-9
    Current loss is: 1.5966756710604791e-9
    Current loss is: 1.587513540657822e-9
    Current loss is: 1.5684930312390978e-9
    Current loss is: 1.5619038531846641e-9
    Current loss is: 1.5190999882273102e-9
    Current loss is: 1.441762382582697e-9
    Current loss is: 1.3964357030953902e-9
    Current loss is: 1.2539764887245526e-9
    Current loss is: 1.1643390288354026e-9
    Current loss is: 1.0691545509380193e-9
    Current loss is: 9.747801956462894e-10
    Current loss is: 8.327160145615407e-10
    Current loss is: 7.858308652406551e-10
    Current loss is: 7.699494394597169e-10
    Current loss is: 7.665995740202504e-10
    Current loss is: 7.598748956182459e-10
    Current loss is: 7.510783902201869e-10
    Current loss is: 7.274605003259455e-10
    Current loss is: 5.942350489312923e-10
    Current loss is: 3.7594824357319163e-10
    Current loss is: 1.9152884481591678e-10
    Current loss is: 1.7084789235739298e-10
    Current loss is: 1.4696629937762685e-10
    Current loss is: 1.2737912225537868e-10
    Current loss is: 1.2225465519282306e-10
    Current loss is: 1.2109879515294232e-10
    Current loss is: 1.1516824432513035e-10
    Current loss is: 1.0612826374939878e-10
    Current loss is: 9.606475441850413e-11
    Current loss is: 9.160603303437834e-11
    Current loss is: 9.016013983803322e-11
    Current loss is: 8.751600985567617e-11
    Current loss is: 8.55686882713354e-11
    Current loss is: 8.323258833851903e-11
    Current loss is: 8.20333978044837e-11
    Current loss is: 8.116050051129906e-11
    Current loss is: 8.038096145727276e-11
    Current loss is: 7.988837874105807e-11
    Current loss is: 7.981566812801984e-11
    Current loss is: 7.960856579915771e-11
    Current loss is: 7.951130337361867e-11
    Current loss is: 7.947851829054564e-11
    Current loss is: 7.944457006042009e-11
    Current loss is: 7.943583486107046e-11
    Current loss is: 7.943461058831474e-11
    127.625623 seconds (155.08 M allocations: 16.388 GiB, 2.42% gc time, 93.34% compilation time: 0% of which was recompilation)





    u: [0mComponentVector{Float64}(x = (layer_1 = (weight = [1.2391641315990887; -1.2624372067060499;;], bias = [-0.005714249217968456; -0.0048969582923349105;;]), layer_2 = (weight = [-0.14549359235307663 0.7801735961325359; -0.2863306790694919 -0.33967704723698366], bias = [0.01236619004069902; 0.006945415265485195;;]), layer_3 = (weight = [-0.99212247353105 0.6823952041473823; -0.6367107980763139 0.9539093184220228], bias = [0.04096491406752025; 0.03209712935620382;;]), layer_4 = (weight = [1.2372094916144774 -1.1183025634061905], bias = [0.08726341504595454;;])), y = (layer_1 = (weight = [1.1719487142387959; 0.25250084267997763;;], bias = [0.004043825564106947; -0.015039295871020238;;]), layer_2 = (weight = [0.3140790405332895 1.1939357277741305; -0.0378375426920909 0.06865211573999874], bias = [0.004343847690013339; -0.024609658027291946;;]), layer_3 = (weight = [-0.0522413572545619 0.5391127428091234; 0.3481422703026555 0.02710704257481156], bias = [0.04447086356319616; 0.016388670109696045;;]), layer_4 = (weight = [0.42094091157073443 0.22531715155912058], bias = [-0.3761239742558119;;])), z = (layer_1 = (weight = [0.18837962443157028; -0.04756636233999905;;], bias = [0.05167050867425198; -0.00801849898765349;;]), layer_2 = (weight = [0.4996432008098123 -0.918676215162128; -0.6718204810312174 0.3515432367747537], bias = [-0.1022323982595707; 0.00870590608207004;;]), layer_3 = (weight = [-0.9272385185912568 -0.8763618639693666; 1.0705309810726864 -0.9755127668954134], bias = [-0.01882043392568643; 0.06330519156929651;;]), layer_4 = (weight = [-0.8898001692561273 -0.20583866053207542], bias = [-0.18423821140462868;;])), σ_ = (constant = [0.05311513042261095;;]), ρ = (constant = [0.18544203917919308;;]), β = (constant = [-0.06726111771137362;;]))




```julia
#Parameters of inversion
print("inversed Parameters",res.u.σ_.constant, res.u.ρ.constant, res.u.β.constant)
```

    inversed Parameters[0.05311513042261095;;][0.18544203917919308;;][-0.06726111771137362;;]


```julia
phi=pinn.phi
θ = res.u
ts=  [0:0.01:1...;;]
x_pred = phi.x(ts, θ.x)
y_pred = phi.x(ts, θ.y)
z_pred = phi.x(ts, θ.z)
```




    1×101 Matrix{Float64}:
     -0.713561  -0.713562  -0.713564  …  -0.713821  -0.713826  -0.71383




```julia
using Plots
Plots.plot(vec(ts), [vec(x_pred),vec(y_pred),vec(z_pred)],  title=["x(t)" "y(t)" "z(t)"])   
```
![image](https://user-images.githubusercontent.com/45444680/193525084-9185953c-aa41-4091-8138-25be6338fb8c.png)







