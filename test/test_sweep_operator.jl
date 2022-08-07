include("../src/sweep_operator.jl")

@testset "Sweep Operator Correctness" begin
    correct_result = [1.1666666666666667 -0.5 -0.0 1.5; -0.5 0.25 -0.0 0.25; 0.0 -0.0 0.16666666666666666 0.3333333333333333; -1.5 -0.25 -0.3333333333333333 3.0833333333333335]
    correct_T1SS = [24.0, 0.25, 0.6666666666666665]
    correct_T2SS = [1.9285714285714284, 0.25, 0.6666666666666666]
    correct_last_see = 3.0833333333
    M  = [  1.   1.   1.   1.
            1.   2.   1.   3.
            1.   3.   1.   3.
            1.   1.  -1.   2.
            1.   2.  -1.   2.
            1.   3.  -1.   1. ]

    M0 = M' * M
    sweepedM0 = sweep_op_full!(M0')
    @test isapprox(correct_result, M0)

    M0 = M' * M
    SSE, TypeISS = sweep_op_fullT1SS!(M0')
    @test isapprox(correct_last_see, SSE)
    @test isapprox(correct_result, M0)
    @test isapprox(correct_T1SS, TypeISS)
    @test isapprox(correct_T2SS, get_TypeIISS(M0))


    A = [0.6694830121789994; 0.21314246720926422; 0.6529660250873977; 0.6385918132605166; 0.037358200179362755; 0.2875097674400453; 0.21811874684632737; 0.4232050450926038; 0.9734084003457419; 0.4895213360877587; 0.22107794000504877;;]
    B = [0.1092789425428552 0.7682468044279679 0.9375459449272519 0.3016008879920138 0.6350731353296186 0.6391931975313466 0.7931988086444425 0.1547756065020972 0.7551009672990262 0.2043390054901021 0.37466999909368104; 0.7518732467227479 0.17404612464595814 0.761770033509471 0.2770864478415873 0.7543141370258291 0.32310676567293395 0.33347421997349236 0.4534191137643454 0.5128610834497598 0.9341505398911903 0.7536289774998648; 0.8217480529578403 0.34093452577281813 0.962711302240167 0.749580537971113 0.6119080109928845 0.4898631756099011 0.22873586091225762 0.7556388096116227 0.5482875796771497 0.3019759415002481 0.24383237803532276; 0.9205908572601128 0.9838016151616238 0.9380559252830013 0.33900574573219244 0.0583887229946799 0.4679092112776474 0.39026963404013393 0.5418546773143993 0.6935042985856147 0.3282403487260446 0.6961565982332839; 0.1414240797001498 0.3412940519296852 0.8408526865689825 0.462294772065576 0.31393723531549245 0.8748575543600328 0.14699174692940986 0.39306278859334387 0.10912929212661415 0.9141524544576278 0.29686130512125486; 0.37906779731717744 0.8475450403582677 0.2038618175296657 0.4683061428672235 0.9297014862588634 0.5466002831094359 0.6751069695895096 0.9743862251742357 0.9564283751044231 0.6116585273083751 0.7909339981249403; 0.5359109396842612 0.012988843952200346 0.9835824622845665 0.8686665313226049 0.45336455279489085 0.5839037022748381 0.9157470648204125 0.7116357064138576 0.5183459854159834 0.08679668241144856 0.2799014486961946; 0.7554124198898702 0.9323865537553858 0.8678647805277917 0.038064063816100724 0.3429878578591379 0.31534343578500357 0.475130227302884 0.36996903727864516 0.030115361864833545 0.34244322477966027 0.4523144307547967; 0.8336756262881949 0.5216976063941485 0.28203838317254926 0.7373358446522794 0.5620614410329805 0.057089029616333886 0.2616440484472927 0.7340293213619165 0.7384112237773132 0.7537099230023115 0.6503159099088369; 0.5551618374419353 0.03118050811467732 0.23476295981821782 0.917221770950081 0.223897520609306 0.5216802455519959 0.365440442059882 0.7394898203765863 0.09923845623220229 0.4348799692321885 0.10475084732823181; 0.9893186829830513 0.5621839305113805 0.278541900532278 0.10896953976431789 0.31531406921972904 0.5869378617853271 0.4820934500666002 0.9326376745113245 0.8775590942514956 0.7887445296205713 0.6274169456020134]
    correct_coefs = [0.8626528683885901; 0.7799120469808962; -0.3648600566802049; 1.0021645622078064; 0.2748318846779194; -0.16712897766519763; 0.09387657451235487; -1.251063803103481; 0.46242701312864226; 0.27040571287723963; -0.8997388730388888;;]
    @test isapprox(sweep_linsolve(B, A), correct_coefs)

end