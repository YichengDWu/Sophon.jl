using Sophon, Random, Lux, NNlib
using Test

rng = Random.default_rng()

@testset "Sophon.jl" begin @testset "layers" begin
    @testset "basic" begin
        @testset "FourierFeature" begin
            f1 = FourierFeature(2, (1 => 4,))
            @test f1.out_dims == 8
            f2 = FourierFeature(2, (1 => 4, 2 => 5))
            ps, st = Lux.setup(rng, f2)
            @test length(st) == (2)
            y, st = f2(rand(Float32, 2, 2), ps, st)
            @test size(y) == (18, 2)
            @test f2.out_dims == 18
        end
        @testset "FullyConnected" begin
            fc = FullyConnected(2, (4,), sin)
            @test fc == Dense(2, 4, sin)
            fc2 = FullyConnected(2, (4, 5, 6), sin)
            @test values(map(x -> x.out_dims, fc2.layers)) == (4, 5, 6)
        end
    end
    @testset "Nets" begin
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
            m = MultiscaleFourier(2)
            ps, st = Lux.setup(rng, m)
            y, st = m(x, ps, st)
            @test size(y) == (1, 5)
        end
    end
end end
