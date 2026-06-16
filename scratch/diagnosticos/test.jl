hist_file = "scratch/test_append.txt"
open(hist_file, "w") do f
    write(f, "header\n")
end

cb = () -> begin
    try
        open(hist_file, "a") do f
            write(f, "append\n")
        end
    catch e
        println("ERROR IN CLOSURE: ", e)
    end
end

cb()
println("FILE CONTENTS:")
println(read(hist_file, String))
