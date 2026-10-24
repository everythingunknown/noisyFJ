include("common.jl")

function references(g, item, path)
    n = length(g.c)
    if item["dense"]
        measured = @timed vec(sum(abs2, inv(Matrix(g.B)); dims=1))
        scores = measured.value
        open(joinpath(path, "exact.csv"), "w") do io
            println(io, "seconds\n", measured.time)
        end
        serialize(joinpath(path, "Exact.jls"), scores)
    else
        scores = zeros(n); r1 = zeros(Int,n); r2 = similar(r1); successor = similar(r1)
        rng = Xoshiro()
        for _ in 1:CONFIG["selection_forests"]÷2
            sample_forest!(rng,g,r1,successor); sample_forest!(rng,g,r2,successor)
            raw_add!(scores,r1,r2)
        end
        scores ./= CONFIG["selection_forests"]÷2
    end
    nodes = partialsortperm(collect(1:n), 1:CONFIG["selected_sources"]; by=i->(-scores[i],i))
    refs = BigFloat[]
    workspace = FJSolveWorkspace(g; history=CONFIG["history"])
    open(joinpath(path, "reference.csv"), "w") do io
        println(io, "node,value,certificate,selection_score")
        for j in nodes
            if item["dense"]
                value, certificate = BigFloat(scores[j]), 0.0
            else
                q = refined_reference(g,j; epsilon=CONFIG["reference_tolerance"],workspace)
                @assert q.converged && q.certificate <= CONFIG["reference_tolerance"]
                value, certificate = q.value, q.certificate
            end
            push!(refs, value)
            println(io, join((j,value,certificate,scores[j]), ',')); flush(io)
        end
    end
    nodes, refs
end

function iterations(g, nodes, refs, path)
    for (method, name) in ((:fixed, "FP"), (:anderson, "AA"))
        workspace = FJSolveWorkspace(g; history=method == :anderson ? CONFIG["history"] : 0)
        b = zeros(length(g.c))
        open(joinpath(path, name * ".csv"), "w") do io
            println(io, "node,value,certificate,seconds,iterations,absolute_error")
            for (j, ref) in zip(nodes, refs)
                measured = @timed query!(g,b,workspace,j,method)
                q = measured.value
                checked = query_certificate(q.x, b-g.B*q.x).certificate
                @assert checked <= CONFIG["query_tolerance"]*(1+1e-12)
                error = Float64(abs(BigFloat(q.value)-ref))
                @assert error <= checked+1e-12
                println(io, join((j,q.value,checked,measured.time,q.iterations,error), ',')); flush(io)
            end
        end
    end
end

function forests(g, item, nodes, refs, path)
    n = length(g.c); GC.gc()
    common = @timed (zeros(Int,n), zeros(Int,n), zeros(Int,n), Xoshiro())
    r1,r2,successor,rng = common.value
    raw_setup = @timed zeros(n)
    refined_setup = @timed ParallelLeftWorkspace(g; workers=CONFIG["workers"])
    raw, refined = raw_setup.value, refined_setup.value
    sampling = 0.0; raw_time = 0.0; refined_time = 0.0
    open(joinpath(path, "forests.csv"), "w") do io
        println(io, "forests,method,setup_s,sampling_s,accumulation_s,normalization_s,total_s,mae,error_sd,mre_percent")
        for t in 1:maximum(CONFIG["forests"])÷2
            stamp = time_ns()
            sample_forest!(rng,g,r1,successor); sample_forest!(rng,g,r2,successor)
            sampling += (time_ns()-stamp)/1e9
            for k in (isodd(t) ? (1,2) : (2,1))
                stamp = time_ns()
                if k == 1
                    raw_add!(raw,r1,r2); raw_time += (time_ns()-stamp)/1e9
                else
                    add_left_parallel!(refined,r1,r2); refined_time += (time_ns()-stamp)/1e9
                end
            end
            if 2t in CONFIG["forests"]
                normalizations = (@timed(raw./t), @timed(reduce_left!(zeros(n),refined)))
                for (k,name) in enumerate(("PFS","FRPFS"))
                    measured = normalizations[k]; values = measured.value
                    @assert all(isfinite,values) && all(>=(0),values)
                    errors = Float64[abs(BigFloat(values[j])-ref) for (j,ref) in zip(nodes,refs)]
                    relative = Float64[abs(BigFloat(values[j])-ref)/ref for (j,ref) in zip(nodes,refs)]
                    setup = common.time+(k == 1 ? raw_setup.time : refined_setup.time)
                    accumulation = k == 1 ? raw_time : refined_time
                    total = setup+sampling+accumulation+measured.time
                    println(io, join((2t,name,setup,sampling,accumulation,measured.time,total,
                                      mean(errors),std(errors),100mean(relative)), ',')); flush(io)
                    serialize(joinpath(path,"$(name)_$(2t).jls"),values)
                    open(joinpath(path,"$(name)_$(2t).csv"),"w") do out
                        println(out,"node,prediction,reference,absolute_error,relative_error")
                        for (j,ref,e,r) in zip(nodes,refs,errors,relative)
                            println(out,join((j,values[j],ref,e,r),','))
                        end
                    end
                end
                println(item["id"], " ", 2t, " forests"); flush(stdout)
            end
        end
    end
end

function benchmark(item)
    path = joinpath(ROOT,"results",item["id"]); mkpath(path)
    isfile(joinpath(path,"benchmark.complete")) && return
    println("Benchmark ",item["id"]); flush(stdout)
    g = load_graph(item)
    nodes,refs = references(g,item,path)
    iterations(g,nodes,refs,path)
    forests(g,item,nodes,refs,path)
    touch(joinpath(path,"benchmark.complete"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    @assert Threads.nthreads(:default) == CONFIG["workers"]
    warmup()
    for item in selected_datasets()
        benchmark(item); GC.gc()
    end
end
