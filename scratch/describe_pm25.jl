using Pkg
Pkg.activate(".")
using JSON, Statistics

function describe_pm25()
    data = JSON.parsefile("datos_oficiales_pm25.json")
    pm = [Float64(d["pm25"]) for d in data]
    println("PM2.5 Statistics:")
    println("  - Count: ", length(pm))
    println("  - Mean: ", mean(pm))
    println("  - StdDev: ", std(pm))
    println("  - Min: ", minimum(pm))
    println("  - Max: ", maximum(pm))
end

describe_pm25()
