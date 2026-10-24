module ParallelLeftForest

using Base.Threads
using ..NoisyFJ: FJGraph
export ParallelLeftWorkspace, add_left_parallel!, reduce_left!, reset!, row_work

mutable struct LeftBlock
    marks::Vector{Int}
    counts::Vector{Float64}
    gamma::Vector{Float64}
    epoch::Int
end

mutable struct ParallelLeftWorkspace
    graph::FJGraph
    c2::Vector{Float64}
    ranges::Vector{UnitRange{Int}}
    blocks::Vector{LeftBlock}
    pairs::Int
end

function row_work(g, rows)
    isempty(rows) && return 0
    length(rows) + 2 * (g.ptr[last(rows)+1] - g.ptr[first(rows)])
end

function ParallelLeftWorkspace(g::FJGraph; workers=5)
    1 <= workers <= Threads.nthreads(:default) ||
        throw(ArgumentError("workers must be between 1 and the default Julia thread count"))
    n=length(g.c)
    prefix=[i + 2*(g.ptr[i+1]-1) for i in 0:n]
    cuts=[0; [searchsortedfirst(prefix,cld(k*prefix[end],workers))-1
              for k in 1:workers-1]; n]
    ranges=[cuts[k]+1:cuts[k+1] for k in 1:workers]
    blocks=[LeftBlock(zeros(Int,n),zeros(n),zeros(n),0) for _ in 1:workers]
    ParallelLeftWorkspace(g,g.c.^2,ranges,blocks,0)
end

function add_block!(b::LeftBlock,g,c2,rows,r1,r2)
    marks=b.marks;counts=b.counts;gamma=b.gamma;epoch=b.epoch
    @inbounds for i in rows
        if g.ptr[i]==g.ptr[i+1]
            gamma[i]+=c2[i]
            continue
        end
        if epoch==typemax(Int)
            fill!(marks,0);epoch=0
        end
        epoch+=1;stamp=epoch;marks[i]=stamp;counts[i]=1.0
        for k in g.ptr[i]:g.ptr[i+1]-1
            j=r1[g.nbr[k]]
            if marks[j]!=stamp
                marks[j]=stamp;counts[j]=0.0
            end
            counts[j]+=g.weight[k]
        end
        gamma[i]+=counts[i]*c2[i]
        for k in g.ptr[i]:g.ptr[i+1]-1
            j=r2[g.nbr[k]]
            if marks[j]==stamp
                gamma[j]+=counts[j]*g.weight[k]*c2[i]
            end
        end
    end
    b.epoch=epoch
    nothing
end

function add_left_parallel!(w::ParallelLeftWorkspace,r1,r2)
    n=length(w.c2)
    length(r1)==length(r2)==n || throw(DimensionMismatch("root-map length"))
    Threads.@threads :static for k in eachindex(w.blocks)
        add_block!(w.blocks[k],w.graph,w.c2,w.ranges[k],r1,r2)
    end
    w.pairs+=1
    nothing
end

function reduce_left!(out,w::ParallelLeftWorkspace;normalize=true)
    n=length(w.c2);p=length(w.blocks)
    length(out)==n || throw(DimensionMismatch("output length"))
    normalize && w.pairs==0 && throw(ArgumentError("no forest pairs accumulated"))
    scale=normalize ? inv(Float64(w.pairs)) : 1.0
    Threads.@threads :static for k in 1:p
        @inbounds for i in fld((k-1)*n,p)+1:fld(k*n,p)
            value=0.0
            for block in w.blocks
                value+=block.gamma[i]
            end
            out[i]=value*scale
        end
    end
    out
end

function reset!(w::ParallelLeftWorkspace)
    for block in w.blocks
        fill!(block.gamma,0.0)
        fill!(block.marks,0)
        block.epoch=0
    end
    w.pairs=0
    w
end
end
