include("common.jl")
include("../src/MarginalAllocation.jl"); using .MarginalAllocation
include("../src/SmallAllocationControls.jl"); using .SmallAllocationControls

function resampling(item; trials=CONFIG["noise_trials"])
    path = joinpath(ROOT,"results",item["id"],"resampling")
    mkpath(path)
    marker = joinpath(path,"complete")
    protocol = "marginal-allocation; trials=$trials; opinions-redrawn"
    if isfile(marker)
        strip(read(marker,String)) == protocol || error("Existing allocation results use another protocol: $path")
        return
    end
    println("Measurement allocation ",item["id"]); flush(stdout)
    g = load_graph(item); n = length(g.c); factor = lu(g.B)
    base = dirname(path)
    gamma = deserialize(joinpath(base,"Exact.jls"))
    sampled = deserialize(joinpath(base,"FRPFS_4000.jls"))
    structural = factor' \ ones(n)
    @assert norm(g.B'*structural-ones(n),Inf) < 1e-10
    methods = ["Degree","Middle","Uniform","Structural","Centrality"]
    weights = Dict("Degree"=>vec(sum(g.A;dims=1)),
                   "Structural"=>structural, "Centrality"=>sampled)
    static = Dict((name,round)=>Int.(marginal_counts(w,n*(round+1)))
                  for (name,w) in weights for round in 1:2)
    for name in keys(weights)
        @assert all(static[name,2] .>= static[name,1])
    end
    rng = Xoshiro()
    errors = zeros(trials,2,length(methods))
    open(joinpath(path,"trials.csv"),"w") do io
        println(io,"trial,method,round,new_budget,cumulative_new_budget,total_squared_error")
        for trial in 1:trials
            s = rand(rng,n); amplitude = min.(s,1 .-s)
            streams = [Float64[] for _ in 1:n]
            extend_streams!(streams,amplitude,ones(Int,n),rng)
            initial = observation_errors(streams,ones(Int,n))
            counts = [ones(Int,n) for _ in methods]
            estimates = [s+initial for _ in methods]
            for round in 1:2
                noise = Vector{Vector{Float64}}(undef,length(methods))
                for (k,name) in enumerate(methods)
                    previous = counts[k]
                    if name == "Uniform"
                        counts[k] = fill(round+1,n)
                    elseif name == "Middle"
                        observed = clamp.(estimates[k],0,1)
                        proxy = min.(observed,1 .-observed).^2 ./ 3
                        counts[k] = extra_counts(proxy,previous,n)
                    else
                        counts[k] = static[name,round]
                    end
                    @assert sum(counts[k]) == n*(round+1) && all(counts[k] .>= previous)
                    extend_streams!(streams,amplitude,counts[k],rng)
                    noise[k] = observation_errors(streams,counts[k])
                    estimates[k] = s+noise[k]
                end
                rhs = hcat(noise...)
                propagated = factor \ rhs
                @assert norm(g.B*propagated-rhs,Inf) < 1e-9
                for (k,name) in enumerate(methods)
                    value = sum(abs2,@view propagated[:,k])
                    errors[trial,round,k] = value
                    println(io,join((trial,name,round,n,round*n,value),','))
                end
            end
        end
    end
    open(joinpath(path,"summary.csv"),"w") do io
        println(io,"method,round,mean_error,error_sd")
        for round in 1:2, (k,name) in enumerate(methods)
            values = errors[:,round,k]
            println(io,join((name,round,mean(values),std(values)),','))
        end
    end
    open(joinpath(path,"expected_risk.csv"),"w") do io
        println(io,"method,round,expected_error")
        for round in 1:2
            println(io,join(("Uniform",round,count_risk(gamma,fill(round+1,n))),','))
            for name in keys(weights)
                println(io,join((name,round,count_risk(gamma,static[name,round])),','))
            end
        end
    end
    write(marker,protocol*"\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    for item in selected_datasets()
        item["dense"] || continue
        resampling(item); GC.gc()
    end
end
