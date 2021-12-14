@testset "ridge regression" begin
    M  = [  1.   1.   1.   1.
            1.   2.   1.   3.
            1.   3.   1.   3.
            1.   1.  -1.   2.
            1.   2.  -1.   2.
            1.   3.  -1.   1. ]
    df = DataFrame(M, [:x0, :x1, :x2, :y])
    
    # Simple
    t_coefs = [1.5454545454545454, 0.22727272727272727, 0.30303030303030304]
    t_mse = 1.03030303030303
    t_rmse = 1.0150384378451045
    t_vifs = [0.0, 0.8264462809917354, 0.8264462809917352]
    t_r2 = 0.2272727272727274

    coefs, mse, rmse, r2, vifs = ridge(@formula(y ~ x1 + x2), df, 0.1)

    @test isapprox(t_coefs, coefs)
    @test isapprox(t_mse, mse)
    @test isapprox(t_rmse, rmse)
    @test isapprox(t_vifs, vifs)
    @test isapprox(t_r2, r2)


    # series of simple ridge regression
    t_coefs = [[1.5, 0.25, 0.3333333333333333], [1.5454545454545454, 0.22727272727272727, 0.30303030303030304], [1.5833333333333333, 0.20833333333333334, 0.2777777777777778]]
    t_mse = [1.0277777777777777, 1.03030303030303, 1.0362654320987652]
    t_rmse = [1.0137937550497031, 1.0150384378451045, 1.0179712334338162]
    t_vifs = [[0.0, 1.0, 0.9999999999999998], [0.0, 0.8264462809917354, 0.8264462809917352], [0.0, 0.6944444444444445, 0.6944444444444445]]
    t_r2 = [0.22916666666666674, 0.2272727272727274, 0.22280092592592604]

    coefs, mse, rmse, r2, vifs = ridge(@formula(y ~ x1 + x2), df, 0:0.1:0.2)
    @test isapprox(t_coefs, coefs)
    @test isapprox(t_mse, mse)
    @test isapprox(t_rmse, rmse)
    @test isapprox(t_vifs, vifs)
    @test isapprox(t_r2, r2)

    # TODO test with weights

    # TODO test without intercept
end
