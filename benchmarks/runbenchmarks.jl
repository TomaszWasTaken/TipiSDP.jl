using TipiSDP

using Quadmath
using DoubleFloats

using Printf

dir = "./problems/SDPLIB-master/data/"
# files = readdir(dir, sort=false)
# filter!(s -> occursin(r"truss", s), files)

files = ["arch8.dat-s", "control1.dat-s", "truss1.dat-s", "theta1.dat-s", "qap6.dat-s", "ss30.dat-s"]



TestFloats = [Float64, Float128]

for T in TestFloats
    dest_file = open("./benchmarks/$T/sdplib_partial.csv", "w")
    println(dest_file, "problem, iters, time, err1, err2, err3, err4, err5")
    for file in files
        prb = TipiSDP.read_from_file(dir*file, T)
        TipiSDP.set_solution_relgap(prb.settings, T(1e-8))
        TipiSDP.set_silent(prb.settings, true)
        t_elapsed = @elapsed TipiSDP.optimize!(prb)
        err1, err2, err3, err4, err5 = TipiSDP.DIMACS_criterion(prb)
        iterCount = prb.iter_count
        println("problem: ", file)
        @printf("err1: %.2e, err2: %.2e, err3: %.2e, err4: %.2e, err5: %.2e\n", err1, err2, err3, err4, err5)
        println("iterCount: ", iterCount)
        println("t: ", t_elapsed)
        println("-------------------------------------------------------------------------------")
        println(dest_file, file, ", ", iterCount, ", ", t_elapsed, ", ", err1, ", ", err2, ", ", err3, ", ", err4, ", ", err5)
    end
    close(dest_file)
end


nothing
