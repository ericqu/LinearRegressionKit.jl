
@testset "from glm" begin 
    fdf = DataFrame(Carb=[0.1,0.3,0.5,0.6,0.7,0.9], OptDen=[0.086,0.269,0.446,0.538,0.626,0.782])
    lm1 = regress(@formula(OptDen ~ 1 + Carb), fdf, req_stats="all")
    target_coefs = [0.005085714285713629, 0.8762857142857156]
    @test isapprox(target_coefs, lm1.coefs)
    @test lm1.p == 2 
    @test isapprox(lm1.R2, 0.9990466748057584)
    @test isapprox(lm1.ADJR2, 0.998808343507198)
    @test isapprox(lm1.AIC, -55.43694668654871) # using the SAS formula rather than the Julia-Statsmodel-GLM
    @test isapprox(lm1.stderrors,  [0.007833679251299831, 0.013534536322659505]) 
    @test isapprox(lm1.t_values, [0.6492114525712505, 64.74442074669666])
    @test isapprox(lm1.p_values, [0.5515952883836446, 3.409192065429258e-7])
    @test isapprox(lm1.p_values, [0.5515952883836446, 3.409192065429258e-7])
    @test isapprox(lm1.ci_low, [-0.016664066127247305, 0.8387078171615459])
    @test isapprox(lm1.ci_up, [0.02683549469867456, 0.9138636114098853])
    @test isapprox(lm1.t_statistic, 2.7764451051977934)
    @test isapprox(lm1.VIF, [0.,  1.])
    @test isapprox(lm1.PRESS, 0.0010755278041106075)
    @test isapprox(lm1.Type1SS, [1.2576681666666665, 0.3135496333333334])
    @test isapprox(lm1.Type2SS, [3.152636815919868e-5, 0.31354963333333363])
end

@testset "from glm regresspredict" begin 
    fdf = DataFrame(Carb=[0.1,0.3,0.5,0.6,0.7,0.9], OptDen=[0.086,0.269,0.446,0.538,0.626,0.782])
    lm1 = regress(@formula(OptDen ~ 1 + Carb), fdf, α=0.05, req_stats=[:default, :aic, :vif, :press, :t1ss, :t2ss])
    target_coefs = [0.005085714285713629, 0.8762857142857156]
    @test isapprox(target_coefs, lm1.coefs)
    @test lm1.p == 2 
    @test isapprox(lm1.R2, 0.9990466748057584)
    @test isapprox(lm1.ADJR2, 0.998808343507198)
    @test isapprox(lm1.AIC, -55.43694668654871) # using the SAS formula rather than the Julia-Statsmodel-GLM
    @test isapprox(lm1.stderrors,  [0.007833679251299831, 0.013534536322659505]) 
    @test isapprox(lm1.t_values, [0.6492114525712505, 64.74442074669666])
    @test isapprox(lm1.p_values, [0.5515952883836446, 3.409192065429258e-7])
    @test isapprox(lm1.p_values, [0.5515952883836446, 3.409192065429258e-7])
    @test isapprox(lm1.ci_low, [-0.016664066127247305, 0.8387078171615459])
    @test isapprox(lm1.ci_up, [0.02683549469867456, 0.9138636114098853])
    @test isapprox(lm1.t_statistic, 2.7764451051977934)
    @test isapprox(lm1.VIF, [0.,  1.])
    @test isapprox(lm1.PRESS, 0.0010755278041106075)
    @test isapprox(lm1.Type1SS, [1.2576681666666665, 0.3135496333333334])
    @test isapprox(lm1.Type2SS, [3.152636815919868e-5, 0.31354963333333363])
 
    lm1 = regress(@formula(OptDen ~ 1 + Carb), fdf, α=0.05, req_stats=["none"])
    target_coefs = [0.005085714285713629, 0.8762857142857156]
    @test isapprox(target_coefs, lm1.coefs)
    @test lm1.p == 2 

    lm1 = regress(@formula(OptDen ~ 1 + Carb), fdf, α=0.05, req_stats=["r2", "P_values"])
    target_coefs = [0.005085714285713629, 0.8762857142857156]
    @test isapprox(target_coefs, lm1.coefs)
    @test lm1.p == 2 
    @test isapprox(lm1.R2, 0.9990466748057584)
    @test isapprox(lm1.p_values, [0.5515952883836446, 3.409192065429258e-7])

    df = DataFrame(y=[1., 3., 3., 2., 2., 1.], x1=[1., 2., 3., 1., 2., 3.], x2=[1., 1., 1., -1., -1., -1.])
    lm2 = regress(@formula(y ~ 1 + x1 + x2), df, req_stats=["r2"])
    @test [1.5, 0.25, 0.3333333333333333] == lm2.coefs
    @test 0.22916666666666663 == lm2.R2

    y = [3.547744106900422, 9.972950249405148, 16.471345464154027, 22.46768807351274, 20.369933318011807, 21.18590757820348, 29.962620198209024, 30.684400502954748, 29.28429078492597, 34.272759386588824, 33.05504692986838, 45.09273876302829, 45.28374262744938, 54.54563960566191, 46.86173948296966, 46.85926120310666, 67.54337216713414, 66.0400205145086, 64.77001443647681, 63.98759256558095, 15.016939388490687, 12.380885701920008, 10.221963402745288, 24.826987790646672, 22.231511892187548, 18.05125492502642, 21.809661866284717, 28.533306224702308, 24.70476800009685, 39.592181057942916, 37.425282624708906, 33.89372811236659, 44.07442335927311, 42.61943719318272, 52.32531447897055, 44.12096163880534, 49.19965880584543, 52.304114239221036, 59.79488104919937, 65.00419916894333, 10.978506745605673, 12.944637189081874, 16.96525561080181, 15.93489355798633, 33.559135307088305, 28.887571687420433, 28.296418324622593, 32.2405215537489, 31.466024917490223, 37.78855308849255, 48.09725215994402, 40.70190542438069, 43.2525436240948, 44.75159242105558, 52.553066965996074, 53.265851121437095, 60.092700204726015, 62.78014347092756, 69.07895282754228, 74.27890668517966, 10.57283140105129, 7.290600131361426, 16.10299402050687, 14.428954773831242, 22.418180226673464, 28.21022852582732, 29.60436622143203, 33.648588929669636, 37.451930576147994, 43.85548900583812, 41.44168340404242, 43.48815266671309, 54.72835160956764, 53.55177468229062, 50.62088969800169, 50.863408563713335, 47.40251347184729, 64.29413401802115, 64.20687412126479, 73.66653216085827, 15.820723594639162, 21.973463928234743, 23.804440715385162, 15.139822408433913, 30.015089890369985, 28.08454394421385, 33.041880065463566, 28.49429418531324, 38.33763660905526, 34.503176013521724, 48.748946870573235, 45.45351516824085, 53.522908096188765, 45.95216131836423, 63.13018379633756, 63.4236036208151, 65.28265579397677, 55.43500146374922, 76.50187470137375, 67.22421998359037, 19.289152114521762, 27.63557010588222, 17.700031078096686, 25.462368278660957, 22.94613580060685, 36.19731649621813, 35.22216995579936, 25.526434727578838, 45.882864925557726, 38.71433797181679, 50.617762276434554, 41.96951039650285, 52.265123328453214, 45.383991138243765, 62.7923270665056, 64.40670612696276, 74.89775274405821, 72.89056716347118, 66.37071343209973, 76.32913721410918, 29.923781174452596, 16.465832617450324, 30.530915733275403, 30.512411156491954, 28.04012351007911, 26.315140074869376, 37.18491928428231, 41.958551085353626, 48.370736387628895, 49.419917216561835, 48.67296268029715, 55.484477112881166, 51.120639229597025, 54.797092949987224, 61.92608065418044, 69.79495109420618, 64.43521939892794, 74.35280205157255, 77.22355341160723, 68.90654715174155, 10.87752158238755, 20.25006748725279, 31.71957876614495, 31.42898857400639, 32.505052996368256, 34.1225641368903, 35.66496173153403, 31.76371648643483, 42.97107454773545, 47.41827763710622]
    x2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    x3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df = DataFrame(y=y, x2=x2, x3=x3)
    lrint = regress(@formula(y ~ x2 +x3), df, req_stats=["default", "pcorr1", "pcorr2", "scorr1", "scorr2", "vif"])


end

@testset "weighted regression" begin 
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
        6.4 14.0  0
    ] # data from https://blogs.sas.com/content/iml/2016/10/05/weighted-regression.html
    
    df = DataFrame(tw, [:y,:x,:w])
    f = @formula(y ~ x)
    lm = regress(f, df, weights="w", req_stats=["default", "t1ss", "t2ss"])

    @test isapprox([2.328237176867885, 0.08535712911515277], lm.coefs)
    @test isapprox(0.014954934572439349, lm.R2)
    @test isapprox(-0.10817569860600562, lm.ADJR2)
    @test isapprox([2.551864989438224, 0.24492357920520605], lm.stderrors)
    @test isapprox([0.3882424860021164, 0.7364546437428148], lm.p_values)
    @test isapprox([10.272024999999998, 0.02220919946016564], lm.Type1SS)
    @test isapprox([0.15221363313265734, 0.022209199460165682], lm.Type2SS)

    res = predict_in_sample(lm, df, req_stats="all")
    @test isapprox([2.9598799323200153, 2.976951358143046, 3.0281656356121376, 3.09645133890426, 3.2159513196654737, 3.326915587515172, 3.326915587515172, 3.335451300426688, 3.386665577895779, 3.4293441424533553], res.predicted)
    @test isapprox([0.2149107957203927, 0.24394048660618184, 0.27451117660671137, 0.22039739135761363, 0.15181541173049415, 0.19864023422787455, 0.19864023422787455, 0.20135117925989024, 0.18147639530421758, 0.1143166949587325], res.leverage)
    @test isapprox([1.957110958921382, 1.765206920504513, 1.4298044633303422, 1.244877562198597, 1.1810280676815839, 1.3571510747551465, 1.3571510747551465, 1.388160919947073, 1.7203164594367346, 2.412834568973736], res.stdi)
    @test isapprox([0.8231374655094712, 0.7816957716988308, 0.6635670917274183, 0.5290286945664605, 0.4287722865207785, 0.5524810400784785, 0.5524810400784785, 0.568305570562634, 0.6742266146509985, 0.7728194750528609], res.stdp)
    @test isapprox([1.5732681689017483, 1.3761754659974743, 1.0787484567368795, 0.9949760929369788, 1.0134771577471209, 1.1096794313749598, 1.1096794313749598, 1.1318340411046413, 1.4318958288930972, 2.151109196482762], res.stdr)
    @test isapprox([0.02407872915271305, 4.5252332002240634e-5, 0.0026705586843115184, 0.414368722166146, 0.31984274220916814, 0.0075059979227238794, 0.10614122824505004, 0.15735260690358094, 0.008083642818377243, 0.054162396113877506], res.cooksd)
    @test isapprox([1.0617215330136984, 1.1743576761339307, 1.4979771781033089, 1.8765089815923404, 2.2272006538876132, 2.0528920244723023, 2.0528920244723023, 2.024936304649691, 1.8318962164458708, 1.6472192372151455], res.lclp)
    @test isapprox([4.858038331626332, 4.779545040152161, 4.5583540931209665, 4.3163936962161795, 4.204701985443334, 4.600939150558042, 4.600939150558042, 4.645966296203685, 4.941434939345688, 5.2114690476915655], res.uclp)
    @test isapprox([-1.5532260320060822, -1.093623100031372, -0.2689693693610047, 0.225758532651414, 0.49249571179955565, 0.19731959703302637, 0.19731959703302637, 0.13434647869991867, -0.5803912914251206, -2.134662351163641], res.lcli)
    @test isapprox([7.472985896646113, 7.047525816317464, 6.325300640585279, 5.967144145157105, 5.939406927531392, 6.456511577997318, 6.456511577997318, 6.536556122153457, 7.353722447216679, 8.993350636070351], res.ucli)
    @test isapprox([-0.41943258330883165, 0.01674833073720578, -0.11880956567004286, 1.7121503453084896, -1.8904731152742318, 0.24609306504533465, -0.9254164387302033, 1.117256288156119, -0.2700375055877384, 0.9161114929771242], res.student)
    @test isapprox([-0.39672963851269843, 0.015666903524005342, -0.1112343500884851, 2.0120993810497283, -2.3774333761244986, 0.23107529120204373, -0.9160676175777195, 1.1376116334771753, -0.25375610179858576, 0.9057709902138592], res.rstudent)
    @test isapprox([-0.8405158658696843, 0.03048522166395697, -0.1766610752356951, 2.1851500267069777, -2.2588848537963426, 0.3407762956775175, -1.2814663667643953, 1.5833601286751433, -0.47239392447269685, 2.225011859577532], res.press)
end

@testset "predictions statistics" begin
    t_carb = [0.1, 0.3, 0.5, 0.6, 0.7, 0.9]
    t_optden = [0.086, 0.269, 0.446, 0.538, 0.626, 0.782]
    t_leverage = [0.5918367346938774, 0.2816326530612245, 0.1673469387755102, 0.1836734693877552, 0.24897959183673485, 0.5265306122448984]
    t_predicted = [0.09271428571428536, 0.26797142857142836, 0.44322857142857136, 0.5308571428571429, 0.6184857142857143, 0.7937428571428573]
    t_residuals = [-0.006714285714285367, 0.0010285714285716563, 0.002771428571428647, 0.0071428571428571175, 0.007514285714285696, -0.011742857142857277]
    t_stdp = [0.0066535244611503905, 0.00458978457544483, 0.0035380151243903845, 0.003706585424647827, 0.004315515434962329, 0.006275706318490394]
    t_stdi = [0.01091189203370195, 0.009791124677432761, 0.009344386069745653, 0.009409504530540022, 0.009665592246181276, 0.010685714285717253]
    t_stdr = [0.00552545131594831, 0.00733033952495041, 0.007891923021648551, 0.007814168189246363, 0.0074950868260910365, 0.005951093194035971]
    t_student = [-1.2151560714878948, 0.1403170242074994, 0.3511727830880084, 0.9140905301586566, 1.0025615297914663, -1.9732268946192373]
    t_rstudent = [-1.3249515797718379, 0.12181828539462737, 0.3089239861926457, 0.8900235681358146, 1.003419757326401, -10.478921731163984]
    t_lcli = [0.06241801648886679, 0.24078690838638886, 0.4172843964641476, 0.5047321700609886, 0.5916497280049665, 0.7640745580187355]
    t_ucli = [0.12301055493970393, 0.2951559487564679, 0.4691727463929951, 0.5569821156532972, 0.6453217005664621, 0.8234111562669791]
    t_lclp = [0.07424114029181057, 0.2552281436530222, 0.43340546665434193, 0.520566011897882, 0.6065039225799076, 0.7763187030532258]
    t_uclp = [0.11118743113676015, 0.2807147134898345, 0.4530516762028008, 0.5411482738164038, 0.630467505991521, 0.8111670112324888]
    t_press = [-0.016449999999999146, 0.001431818181818499, 0.0033284313725491102, 0.00874999999999997, 0.010005434782608673, -0.02480172413793134]
    t_cooksd = [1.0705381016035724, 0.0038594654616162225, 0.012392684477580572, 0.09400066844914513, 0.16661116000566892, 2.1649894168822432]


    fdf = DataFrame([[0.1,0.3,0.5,0.6,0.7,0.9],[0.086,0.269,0.446,0.538,0.626,0.782]], [:Carb, :OptDen])
    lm1 = regress(@formula(OptDen ~ 1 + Carb), fdf)
    results = predict_in_sample(lm1, fdf, α=0.05, req_stats=["all"])
    @test isapprox(t_leverage, results.leverage)
    @test isapprox(t_predicted, results.predicted) 
    @test isapprox(t_residuals, results.residuals)
    @test isapprox(t_stdp, results.stdp)
    @test isapprox(t_stdi, results.stdi)
    @test isapprox(t_stdr, results.stdr)
    @test isapprox(t_student, results.student)
    @test isapprox(t_rstudent, results.rstudent)
    @test isapprox(t_lcli, results.lcli)
    @test isapprox(t_ucli, results.ucli)
    @test isapprox(t_lclp, results.lclp)
    @test isapprox(t_uclp, results.uclp)
    @test isapprox(t_press, results.press)
    @test isapprox(t_cooksd, results.cooksd)

    results = predict_in_sample(lm1, fdf, α=0.05, req_stats=["none"])
    @test isapprox(t_predicted, results.predicted) 
    @test_throws ArgumentError("column name :leverage not found in the data frame") t_leverage == results.leverage 

    results = predict_out_of_sample(lm1, fdf)
    @test isapprox(t_predicted, results.predicted)
    @test_throws ArgumentError("column name :leverage not found in the data frame") t_leverage == results.leverage 

end
