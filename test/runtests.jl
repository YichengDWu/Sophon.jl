using Sophon, Random, Lux, NNlib
using Test

rng = Random.default_rng()

@testset "Sophon.jl" begin @testset "layers" begin @testset "basic" begin
    @testset "FourierFeature" begin
        @test_throws AssertionError FourierFeature(2, 3)
        f1 = FourierFeature(2, 4)
        @test f1.num_modes == 2
        f2 = FourierFeature(2; num_modes=2)
        @test f2.out_dims == 4
        f3 = FourierFeature(2 => 4)
        @test f3.out_dims == 4
        @test_throws MethodError FourierFeature(2 => 4; num_modes=2, std=10.0)
        f4 = FourierFeature(2 => 4; std=1.0)
        ps, st = Lux.setup(rng, f4)
        @test size(st.modes) == (2, 2)
        y, st = f4(rand(Float32, 2, 2), ps, st)
        @test size(y) == (4, 2)
    end
    @testset "PINNAttention" begin
        x = rand(Float32, 3, 5)
        layers = Chain(Dense(4, 4, relu), Dense(4, 4, relu), Dense(4, 4, relu))
        m = PINNAttention(Dense(3, 4, relu), Dense(3, 4, relu), Dense(3, 4, relu);
                             fusion_layers=layers)
        ps, st = Lux.setup(rng, m)
        y, st = m(x, ps, st)
        @test size(y) == (4, 5)

        m2 = PINNAttention(3, 4, relu; fusion_layers=layers)
        ps2, st2 = Lux.setup(rng, m2)
        y2, st2 = m2(x, ps2, st2)
        @test size(y2) == (4, 5)
    end
    @testset "MultiscaleFourier" begin
        x = rand(Float32, 2, 5)
        chain = Chain(Dense(4, 4, relu), Dense(4, 4, relu), Dense(4, 4, relu))
        m = MultiscaleFourier(2,1,4,chain; std = [1,20,50])
        ps, st = Lux.setup(rng, m)
        y, st = m(x, ps, st)
        @test size(y) == (1, 5)
    end
end end end
