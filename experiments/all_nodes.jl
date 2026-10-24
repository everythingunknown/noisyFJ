include("common.jl")

function all_nodes(item)
    item["dense"] || return
    path = joinpath(ROOT,"results",item["id"])
    exact = deserialize(joinpath(path,"Exact.jls"))
    estimate = deserialize(joinpath(path,"FRPFS_4000.jls"))
    n = length(exact)
    @assert length(estimate) == n && all(>(0),exact)
    order = sortperm(1:n;by=i->(-exact[i],i))
    relative = abs.(estimate-exact)./exact
    open(joinpath(path,"all_nodes.csv"),"w") do io
        println(io,"rank,node,exact,frpfs,signed_error,relative_error")
        for (rank,j) in enumerate(order)
            println(io,join((rank,j,exact[j],estimate[j],estimate[j]-exact[j],relative[j]),','))
        end
    end
    top = order[1:cld(n,10)]
    open(joinpath(path,"all_nodes_summary.csv"),"w") do io
        println(io,"nodes,forests,mae,mre_percent,top10_mre_percent,relative_l2_percent")
        println(io,join((n,4000,mean(abs.(estimate-exact)),100mean(relative),
                        100mean(relative[top]),100norm(estimate-exact)/norm(exact)),','))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    for item in selected_datasets()
        all_nodes(item)
    end
end
