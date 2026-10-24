include("common.jl")
include("../src/KrylovBaseline.jl"); using .KrylovBaseline
include("../src/Wang2025SOR.jl"); using .Wang2025SOR

const SOLVER_PARAMETERS = TOML.parsefile(joinpath(ROOT,"solver_parameters.toml"))
const BASELINE_METHODS = ["AA","GMRES","SOR"]

function baseline_query(g,b,w,name,j,omega)
    if name == "SOR"
        q = sor_gamma!(w,j;epsilon=1e-8,omega,shift=1e-5,sigma=1e-5,
                       maxpush=1000length(b),check_every=length(b))
        return (x=q.x,converged=q.converged,iterations=0,pushes=q.pushes)
    elseif name == "GMRES"
        q = gmres!(g,b,w;stop=(x,r)->query_certificate(x,r).certificate<=1e-8,
                   restart=30,maxiter=20000)
        return (x=q.x,converged=q.converged,iterations=q.iterations,pushes=0)
    end
    q = solve_fj(g,b;method=:anderson,history=3,workspace=w,maxiter=20000,
                 stop_test=(x,r,c)->query_certificate(x,r;scale=c).certificate<=1e-8)
    (x=q.x,converged=q.converged,iterations=q.iterations,pushes=0)
end

function baseline_warmup()
    g = FJGraph(sparse([0. .3 2;0 0 0;.7 0 0])); b = zeros(3)
    ws = Dict("AA"=>FJSolveWorkspace(g;history=3),
              "GMRES"=>KrylovWorkspace(3),"SOR"=>SORWorkspace(g))
    for j in 1:3, name in BASELINE_METHODS
        fill!(b,0); b[j] = 1
        q = baseline_query(g,b,ws[name],name,j,1.2)
        @assert q.converged && abs(sum(abs2,q.x)-sum(abs2,g.B\b))<=1e-8
    end
end

function solver_baselines(item)
    id = item["id"]; haskey(SOLVER_PARAMETERS,id) || return
    path = joinpath(ROOT,"results",id,"solver_baselines"); mkpath(path)
    isfile(joinpath(path,"complete")) && return
    println("Solver baselines ",id); flush(stdout)
    GC.gc()
    loaded = @timed load_graph(item); g = loaded.value; n = length(g.c)
    buffers = @timed (zeros(n),zeros(n)); b,residual = buffers.value
    pars = SOLVER_PARAMETERS[id]; omega = pars["omega"]
    training = deserialize(joinpath(dirname(path),"PFS_4000.jls"))
    nodes = collect(partialsortperm(1:n,1:8;by=i->(-training[i],i)))
    rng = Xoshiro(); seen = Set(nodes)
    while length(nodes)<16
        j = rand(rng,1:n)
        if !(j in seen)
            push!(nodes,j); push!(seen,j)
        end
    end
    exact = item["dense"] ? deserialize(joinpath(dirname(path),"Exact.jls")) : nothing
    reference_workspace = exact===nothing ? FJSolveWorkspace(g;history=3) : nothing
    refs = BigFloat[]; refcerts = Float64[]
    open(joinpath(path,"references.csv"),"w") do io
        println(io,"node,reference,certificate")
        for j in nodes
            q = exact===nothing ? refined_reference(g,j;epsilon=1e-12,workspace=reference_workspace) :
                (value=BigFloat(exact[j]),certificate=0.0)
            @assert q.certificate<=1e-12
            push!(refs,q.value); push!(refcerts,q.certificate)
            println(io,join((j,q.value,q.certificate),','))
        end
    end
    reference_workspace = nothing; GC.gc()
    setup = Dict{String,Any}()
    for name in BASELINE_METHODS
        setup[name] = @timed (name=="AA" ? FJSolveWorkspace(g;history=3) :
                              name=="GMRES" ? KrylovWorkspace(n;restart=30) : SORWorkspace(g))
    end
    times = Dict((name,j)=>Float64[] for name in BASELINE_METHODS for j in nodes)
    open(joinpath(path,"queries.csv"),"w") do io
        println(io,"node,query_group,repeat,method,seconds,certificate,value,absolute_error,converged")
        for (k,j) in enumerate(nodes)
            fill!(b,0); b[j] = 1
            for rep in 1:3, name in circshift(BASELINE_METHODS,mod(k+rep,3))
                GC.gc()
                measured = @timed baseline_query(g,b,setup[name].value,name,j,omega)
                q = measured.value
                mul!(residual,g.B,q.x); @. residual=b-residual
                cert = query_certificate(q.x,residual).certificate
                value = sum(abs2,q.x); error = Float64(abs(BigFloat(value)-refs[k]))
                valid = q.converged && cert<=1e-8*(1+1e-10) && error<=cert+refcerts[k]+2e-12
                println(io,join((j,k<=8 ? "screened" : "uniform",rep,name,measured.time,
                                 cert,value,error,valid),','))
                @assert valid
                push!(times[name,j],measured.time)
            end
        end
    end
    open(joinpath(path,"summary.csv"),"w") do io
        println(io,"method,omega,graph_setup_s,workspace_setup_s,retained_tuning_s,query_batch_s,total_s")
        for name in BASELINE_METHODS
            graph_setup = loaded.time+buffers.time
            tuning = name=="SOR" ? pars["retained_tuning_seconds"] : 0.0
            query_time = sum(median(times[name,j]) for j in nodes)
            total = graph_setup+setup[name].time+tuning+query_time
            println(io,join((name,name=="SOR" ? omega : "",graph_setup,setup[name].time,
                             tuning,query_time,total),','))
        end
    end
    open(joinpath(path,"protocol.toml"),"w") do io
        TOML.print(io,Dict("sources"=>nodes,"screened_sources"=>8,"uniform_sources"=>8,
            "repeats"=>3,"julia_threads"=>Threads.nthreads(),"blas_threads"=>BLAS.get_num_threads(),
            "omega"=>omega,"retained_tuning_seconds"=>pars["retained_tuning_seconds"],
            "output_tolerance"=>1e-8,"preprocessing_included"=>true))
    end
    touch(joinpath(path,"complete"))
end

if abspath(PROGRAM_FILE)==@__FILE__
    @assert Threads.nthreads()==1 && BLAS.get_num_threads()==1
    baseline_warmup()
    for item in selected_datasets()
        solver_baselines(item); GC.gc()
    end
end
