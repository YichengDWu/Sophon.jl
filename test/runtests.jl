using Sophon, Random, Lux, NNlib
using Test

rng = Random.default_rng()

@testset "Sophon.jl" begin @testset "layers" begin
    @testset "basic" begin
        @testset "FourierFeature" begin
            f1 = FourierFeature(2, (1 => 4,))
            @test f1.out_dim == 8
            f2 = FourierFeature(2, (1 => 4, 2 => 5))
            ps, st = Lux.setup(rng, f2)
            @test length(st) == (2)
            y, st = f2(rand(Float32, 2, 2), ps, st)
            @test size(y) == (18, 2)
            @test f2.out_dim == 18
        end
        @testset "FullyConnected" begin
            fc = FullyConnected(2, (4,), sin)
            @test fc == Dense(2, 4, sin)
            fc2 = FullyConnected(2, (4, 5, 6), sin)
            @test values(map(x -> x.out_dims, fc2.layers)) == (4, 5, 6)
            @test fc2.layers[end].activation == identity
            fc3 = FullyConnected(2, 4, 3, sin)
            @test values(map(x -> x.out_dims, fc3.layers)) == (4, 4, 4)
            @test fc3.layers[end].activation == identity

            fc4 = FullyConnected(2, (4, 5, 6), sin; use_activation = true)
            @test fc4.layers[end].activation == sin
            fc5 = FullyConnected(2, 4, 5, sin; use_activation = true)
            @test fc5.layers[end].activation == sin
        end
        @testset "Sine" begin
            # first layer
            s = Sine(2, 3; is_first = true)
            x = rand(Float32, 2, 5)
            ps, st = Lux.setup(rng, s)
            @test st.omega isa AbstractFloat
            y, st = s(x, ps, st)
            @test size(y) == (3, 5)

            # hidden layer
            s2 = Sine(2, 3)
            ps2, st2 = Lux.setup(rng, s2)
            @test st2 == NamedTuple()
            y2, st2 = s2(x, ps2, st2)
            @test size(y2) == (3, 5)

            # last layer
            s3 = Sine(2, 3, identity)
            ps3, st3 = Lux.setup(rng, s3)
            @test st3 == NamedTuple()
            y3, st3 = s3(x, ps3, st3)
            @test size(y3) == (3, 5)
        end
    end
    @testset "Nets" begin
        @testset "PINNAttention" begin
            x = rand(Float32, 3, 5)
            layers = Chain(Dense(4, 4, relu), Dense(4, 4, relu), Dense(4, 4, relu))
            m = PINNAttention(Dense(3, 4, relu), Dense(3, 4, relu), Dense(3, 4, relu),
                              layers)
            ps, st = Lux.setup(rng, m)
            y, st = m(x, ps, st)
            @test size(y) == (4, 5)

            m3 = PINNAttention(3, 4, 3, relu)
            ps3, st3 = Lux.setup(rng, m3)
            y3, st3 = m3(x, ps3, st3)
            @test size(y3) == (4, 5)
        end
        @testset "MultiscaleFourier" begin
            x = rand(Float32, 2, 5)
            m = MultiscaleFourier(2)
            ps, st = Lux.setup(rng, m)
            y, st = m(x, ps, st)
            @test size(y) == (1, 5)
        end
        @testset "FourierAttention" begin
            fa = FourierAttention(2, 4, 3, Lux.relu; modes=(1 => 2, 10 => 3))
            x = rand(Float32, 2, 5)
            ps, st = Lux.setup(rng, fa)
            y, st = fa(x, ps, st)
            @test size(y) == (4, 5)
        end
    end
end end
