using TipiSDP

using Quadmath
using DoubleFloats

dir = "./problems/SDPLIB-master/data/"
files = readdir(dir, sort=false)
filter!(s -> occursin(r"arch", s), files)

TestFloats = [Float64, Double64, Float128]

for T in TestFloats
    for file in files
        prb = TipiSDP.read_from_file(dir*file, T)
        TipiSDP.optimize!(prb)
    end
end

nothing
