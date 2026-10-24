module MarginalAllocation
export marginal_counts, count_risk

function marginal_counts(scores,budget)
    n=length(scores);budget>=n || throw(ArgumentError("one initial observation per node"))
    all(x->isfinite(x)&&x>=0,scores) || throw(ArgumentError("nonnegative finite scores"))
    counts=ones(Int32,n);budget==n && return counts
    maximum(scores)>0 || throw(ArgumentError("positive score required"))
    # The upper endpoint must be feasible even when several nodes tie for
    # the largest first gain and the budget contains fewer extra observations.
    lo=0.0;hi=maximum(scores)
    function atthreshold(w,lambda)
        w==0 ? 1 : max(1,1+floor(Int,(sqrt(1+4w/lambda)-1)/2))
    end
    for _ in 1:55
        lambda=(lo+hi)/2
        total=0
        @inbounds for w in scores
            total+=atthreshold(w,lambda)
            total>budget && break
        end
        if total>budget;lo=lambda;else;hi=lambda;end
    end
    @inbounds for i in eachindex(scores)
        counts[i]=atthreshold(scores[i],hi)
    end
    remaining=budget-sum(counts)
    @assert 0<=remaining<=n
    if remaining>0
        order=partialsortperm(1:n,1:remaining;
            by=i->(-scores[i]/(Float64(counts[i])*(counts[i]+1)),i))
        counts[order].+=1
    end
    @assert sum(counts)==budget && minimum(counts)>=1
    last=Inf;next=0.0
    @inbounds for i in eachindex(scores)
        k=Float64(counts[i]);next=max(next,scores[i]/(k*(k+1)))
        k>1 && (last=min(last,scores[i]/((k-1)*k)))
    end
    @assert last+1e-11*maximum(scores)>=next
    counts
end

function count_risk(gamma,counts)
    total=0.0
    @inbounds @simd for i in eachindex(gamma,counts)
        total+=gamma[i]/counts[i]
    end
    total/36
end
end
