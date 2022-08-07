using Sophon, Random, Lux
using Test

rng = Random.default_rng()

@testset "Sophon.jl" begin
    @testset "layers" begin
        @testset "basic" begin
            @testset "FourierFeature" begin
                @test_throws AssertionError FourierFeature(2,3)
                f1 = FourierFeature(2,4)
                @test f1.num_modes == 2
                f2 = FourierFeature(2; num_modes=2)
                @test f2.out_dims == 4
                f3 = FourierFeature(2 => 4)
                @test f3.out_dims == 4
                @test_throws MethodError FourierFeature(2 => 4; num_modes=2, std = 10.0)
                f4 = FourierFeature(2 => 4; std = 1.0)
                ps, st = Lux.setup(rng, f4)
                @test size(ps.modes) == (2,2)
                y, st = f4(rand(Float32, 2, 2), ps, st)
                @test size(y) == (4,2)
            end
        end
    end
end
