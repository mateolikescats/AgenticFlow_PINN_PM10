using Pkg
Pkg.activate(".")
using Downloads
using JSON

function get_elevations(lats::Vector{Float64}, lons::Vector{Float64})
    lat_str = join(lats, ",")
    lon_str = join(lons, ",")
    url = "https://elevation-api.open-meteo.com/v1/elevation?latitude=$lat_str&longitude=$lon_str"
    try
        tmp_file = Downloads.download(url)
        content = read(tmp_file, String)
        data = JSON.parse(content)
        return get(data, "elevation", Float64[])
    catch e
        println("Error fetching elevation: ", e)
        return Float64[]
    end
end

lats = [6.1856666, 6.1686831]
lons = [-75.5972061, -75.5819702]
elevations = get_elevations(lats, lons)
println("SUCCESS_ELEVATIONS: ", elevations)
