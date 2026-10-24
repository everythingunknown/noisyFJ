using LinearAlgebra, SparseArrays, Random, Statistics, Serialization, TOML
include("../src/NoisyFJ.jl"); using .NoisyFJ
include("../src/ParallelLeftForest.jl"); using .ParallelLeftForest
include("../src/MixedReference.jl"); using .MixedReference
BLAS.set_num_threads(1)
const ROOT = normpath(joinpath(@__DIR__, ".."))
const CONFIG = TOML.parsefile(joinpath(ROOT, "datasets.toml"))

function load_graph(item)
    n, m = item["n"], item["m"]
    path = joinpath(ROOT, "data", item["id"])
    source = Vector{Int32}(undef, m); target = similar(source)
    read!(joinpath(path, "sources.i32"), source)
    read!(joinpath(path, "targets.i32"), target)
    g = FJGraph(sparse(Int.(source), Int.(target), ones(m), n, n))
    @assert length(g.c) == n && nnz(g.A) == m
    g
end

function raw_add!(gamma, r1, r2)
    @inbounds for i in eachindex(r1)
        r1[i] == r2[i] && (gamma[r1[i]] += 1.0)
    end
end

function query!(g, b, workspace, j, method)
    fill!(b, 0); b[j] = 1.0; certificate = Ref(Inf)
    function stop(x, r, c)
        certificate[] = query_certificate(x, r; scale=c).certificate
        certificate[] <= CONFIG["query_tolerance"]
    end
    result = solve_fj(g, b; method, workspace,
        history=method == :anderson ? CONFIG["history"] : 0, stop_test=stop)
    @assert result.converged
    (value=sum(abs2, result.x), certificate=certificate[],
     iterations=result.iterations, x=result.x)
end

function warmup()
    g = FJGraph(sparse([0. 0.3 2; 0 0 0; 0.7 0 0]))
    w = ParallelLeftWorkspace(g; workers=CONFIG["workers"])
    r1 = [1,2,1]; r2 = [3,2,3]; raw = zeros(3)
    for _ in 1:30
        raw_add!(raw, r1, r2)
        add_left_parallel!(w, r1, r2)
        reduce_left!(zeros(3), w)
    end
    sample_forest!(Xoshiro(), g, r1, zeros(Int, 3))
    inv(Matrix(g.B))
    refined_reference(g, 1)
    for method in (:fixed, :anderson)
        query!(g, zeros(3), FJSolveWorkspace(g; history=method == :anderson ? 3 : 0), 1, method)
    end
end

function selected_datasets(args=ARGS)
    ids = Set(item["id"] for item in CONFIG["datasets"])
    all(id -> id in ids, args) || error("Unknown dataset")
    [item for item in CONFIG["datasets"] if isempty(args) || item["id"] in args]
end
