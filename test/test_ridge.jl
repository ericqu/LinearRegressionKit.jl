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

    # coefs, mse, rmse, r2, vifs = ridge(@formula(y ~ x1 + x2), df, 0.1)
    rr = ridge(@formula(y ~ x1 + x2), df, 0.1)

    @test isapprox(t_coefs, rr.coefs)
    @test isapprox(t_mse, rr.MSE)
    @test isapprox(t_rmse, rr.RMSE)
    @test isapprox(t_vifs, rr.VIF)
    @test isapprox(t_r2, rr.R2)

    # series of simple ridge regression
    t_intercepts = [1.5, 1.5454545454545454, 1.5833333333333333]
    t_x1s = [0.25, 0.22727272727272727, 0.20833333333333334]
    t_x2s = [0.3333333333333333, 0.30303030303030304, 0.2777777777777778]
    t_vifintercepts = [0.0, 0.0, 0.0]
    t_vifx1s = [1.0, 0.8264462809917354, 0.6944444444444445]
    t_vifx2s = [0.9999999999999998, 0.8264462809917352, 0.6944444444444445]
    t_mse = [1.0277777777777777, 1.03030303030303, 1.0362654320987652]
    t_rmse = [1.0137937550497031, 1.0150384378451045, 1.0179712334338162]
    t_r2 = [0.22916666666666674, 0.2272727272727274, 0.22280092592592604]
    t_adjr2 = [-0.2847222222222221, -0.2878787878787876, -0.2953317901234567]
    
    rdf = ridge(@formula(y ~ x1 + x2), df, 0:0.1:0.2)
    coefs_names = ["(Intercept)",  "x1" , "x2" ]
    vifs_names = "vif_" .* coefs_names
    @test isapprox(t_intercepts, rdf[!, coefs_names[1]])
    @test isapprox(t_x1s, rdf[!, coefs_names[2]])
    @test isapprox(t_x2s, rdf[!, coefs_names[3]])
    @test isapprox(t_vifintercepts, rdf[!, vifs_names[1]])
    @test isapprox(t_vifx1s, rdf[!, vifs_names[2]])
    @test isapprox(t_vifx2s, rdf[!, vifs_names[3]])
    @test isapprox(t_mse, rdf[!, "MSE"])
    @test isapprox(t_rmse, rdf[!, "RMSE"])
    @test isapprox(t_r2, rdf[!, "R2"])
    @test isapprox(t_adjr2, rdf[!, "ADJR2"])

    # test weighted ridge regression
    using LinearRegressionKit
    using Test, DataFrames, StatsModels
    tw = [
        2.3  7.4  0.058 
        3.0  7.6  0.073 
        2.9  8.2  0.114 
        4.8  9.0  0.144 
        1.3 10.4  0.151 
        3.6 11.7  0.119 
        2.3 11.7  0.119 
        4.6 11.8  0.114 
        3.0 12.4  0.073 
        5.4 12.9  0.035 
        12.  11.  -0.1
    ] 

    df = DataFrame(tw, [:y,:x,:w])
    f = @formula(y ~ x)
    lm = regress(f, df, weights="w", req_stats=["default", "vif"])
    rr = ridge(f, df, 0., weights="w")
    @test isapprox(lm.coefs, rr.coefs)
    @test isapprox(lm.VIF, rr.VIF)
    @test isapprox(lm.R2, rr.R2)
    @test isapprox(lm.ADJR2, rr.ADJR2)
    @test isapprox(lm.MSE, rr.MSE)
    @test isapprox(lm.RMSE, rr.RMSE)

    wrdf = ridge(f, df, 0:0.1:0.2, weights="w")
    @test isapprox(0.:0.1:0.2 , wrdf.λ)
    @test isapprox([0.1828582250674794, 0.18288116845535146, 0.18293534034338269], wrdf.MSE)
    @test isapprox([0.42761925245185045, 0.42764607849874114, 0.42770941109985255], wrdf.RMSE)
    @test isapprox([0.014954934572438905, 0.014831340071840282, 0.014539519723204886], wrdf.R2)
    @test isapprox([-0.10817569860600629, -0.10831474241917971, -0.10864304031139449], wrdf.ADJR2)
    @test isapprox([2.3282371768678907, 2.4079428880617186, 2.4743643140565754], wrdf[!, "(Intercept)"])
    @test isapprox([0.08535712911515224, 0.07759739010468385, 0.07113094092929353], wrdf[!, "x"])
    @test isapprox([0. , 0. , 0.], wrdf[!, "vif_(Intercept)"])
    @test isapprox([1.0, 0.8264462809917354, 0.6944444444444445], wrdf[!, "vif_x"])

end
