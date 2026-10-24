module NoisyFJ
using LinearAlgebra, SparseArrays, Random
export FJGraph, FJSolveWorkspace, solve_fj, sample_forest!, query_certificate
include("ErrorCertificates.jl")

struct FJGraph
    A::SparseMatrixCSC{Float64,Int}
    B::SparseMatrixCSC{Float64,Int}
    degree::Vector{Float64}
    c::Vector{Float64}
    ptr::Vector{Int}
    nbr::Vector{Int}
    weight::Vector{Float64}
    cumulative::Vector{Float64}
    unweighted::Bool
end
function FJGraph(A::SparseMatrixCSC)
    A=SparseMatrixCSC{Float64,Int}(A)
    size(A,1)==size(A,2) || throw(ArgumentError("square adjacency required"))
    any(x->!isfinite(x)||x<0,nonzeros(A)) && throw(ArgumentError("nonnegative finite weights required"))
    for i in axes(A,1); A[i,i]=0; end
    dropzeros!(A)
    n=size(A,1); d=vec(sum(A,dims=2)); c=1.0 ./ (1.0 .+ d)
    B=spdiagm(0=>1.0 .+ d)-A
    At=sparse(transpose(A)); ptr=copy(At.colptr); nbr=copy(At.rowval)
    w=copy(At.nzval); cumulative=similar(w)
    for i in 1:n
        total=0.0
        for k in ptr[i]:ptr[i+1]-1; total+=w[k]; cumulative[k]=total; end
    end
    FJGraph(A,B,d,c,ptr,nbr,w,cumulative,all(==(1.0),w))
end
struct FJSolveWorkspace
    graph::FJGraph
    history::Int
    x::Vector{Float64}
    gx::Vector{Float64}
    fx::Vector{Float64}
    DF::Matrix{Float64}
    DG::Matrix{Float64}
    previous_f::Vector{Float64}
    previous_g::Vector{Float64}
    candidate::Vector{Float64}
    work::Vector{Float64}
    gram::Matrix{Float64}
    small::Matrix{Float64}
    rhs::Vector{Float64}
    theta::Vector{Float64}
    q::Float64
end
function FJSolveWorkspace(g;history=3)
    history>=0||throw(ArgumentError("nonnegative history required"))
    n=length(g.degree);h=history;k=h>0 ? n : 0
    FJSolveWorkspace(g,h,zeros(n),Vector{Float64}(undef,n),Vector{Float64}(undef,n),
        Matrix{Float64}(undef,n,h),Matrix{Float64}(undef,n,h),
        Vector{Float64}(undef,k),Vector{Float64}(undef,k),
        Vector{Float64}(undef,k),Vector{Float64}(undef,k),
        zeros(h,h),Matrix{Float64}(undef,h,h),zeros(h),zeros(h),
        h>0 ? maximum(i->g.degree[i]*g.c[i],1:n) : 0.0)
end

function solve_fj(g,b;transposed=false,method=:anderson,tol=1e-10,
                  maxiter=20000,history=3,ridge=1e-10,stop_test=nothing,
                  safeguard=1.0,acceleration_every=1,workspace=nothing)
    method in (:fixed,:anderson) || throw(ArgumentError("unknown method $method"))
    history>=0 && acceleration_every>=1 || throw(ArgumentError("invalid Anderson history/interval"))
    ridge>0 && safeguard>0 || throw(ArgumentError("positive regularization and safeguard required"))
    n=length(b);M=transposed ? transpose(g.A) : g.A
    accelerated=method==:anderson && history>0
    h=accelerated ? history : 0
    ws=isnothing(workspace) ? FJSolveWorkspace(g;history=h) : workspace
    ws.graph===g && ws.history>=h || throw(ArgumentError("workspace graph/history mismatch"))
    x=ws.x;gx=ws.gx;fx=ws.fx;fill!(x,0.0)
    DF=ws.DF;DG=ws.DG;previous_f=ws.previous_f;previous_g=ws.previous_g
    candidate=ws.candidate;work=ws.work
    gram=ws.gram;small=ws.small;rhs=ws.rhs;theta=ws.theta;q=ws.q
    count=0;slot=0;accepted=0;rejected=0;res=Inf
    function apply!(out,z)
        mul!(out,M,z)
        @inbounds @simd for i in 1:n;out[i]=g.c[i]*(b[i]+out[i]);end
        out
    end
    function residual_norm(f)
        if transposed
            a=0.0
            @inbounds @simd for i in 1:n;a+=abs(f[i])/g.c[i];end
            a
        else
            a=0.0
            @inbounds @simd for i in 1:n;a=max(a,abs(f[i])/g.c[i]);end
            a
        end
    end
    function fixed_norm(f)
        transposed ? residual_norm(f) : norm(f,Inf)
    end
    apply!(gx,x)
    for t in 0:maxiter
        @inbounds @simd for i in 1:n;fx[i]=gx[i]-x[i];end
        res=residual_norm(fx)
        if isnothing(stop_test) ? res<=tol : stop_test(x,fx,g.c)
            mul!(fx,transposed ? transpose(g.B) : g.B,x)
            @inbounds @simd for i in 1:n;fx[i]=g.c[i]*(b[i]-fx[i]);end
            res=residual_norm(fx)
            if isnothing(stop_test) ? res<=tol : stop_test(x,fx,g.c)
                return (x=x,iterations=t,residual=res,converged=true,accepted=accepted,rejected=rejected)
            end
            @inbounds @simd for i in 1:n;gx[i]=x[i]+fx[i];end
        end
        t==maxiter && break
        if accelerated
            if t>0
                slot=mod1(slot+1,h);count=min(count+1,h)
                @inbounds @simd for i in 1:n
                    DF[i,slot]=fx[i]-previous_f[i]
                    DG[i,slot]=gx[i]-previous_g[i]
                end
                ds=@view DF[:,slot]
                for k in 1:count
                    value=dot(ds,@view DF[:,k]);gram[k,slot]=gram[slot,k]=value
                end
            end
            copyto!(previous_f,fx);copyto!(previous_g,gx)
        end
        use_acceleration=false
        if accelerated && count>0 && t%acceleration_every==0
            scale=0.0
            for k in 1:count;scale+=gram[k,k];end
            lambda=ridge*max(scale/count,eps())
            @views small[1:count,1:count].=gram[1:count,1:count]
            for k in 1:count;small[k,k]+=lambda;end
            mul!(@view(rhs[1:count]),transpose(@view(DF[:,1:count])),fx)
            copyto!(@view(theta[1:count]),@view(rhs[1:count]))
            factor=cholesky!(Symmetric(@view(small[1:count,1:count]));check=false)
            if issuccess(factor)
                ldiv!(factor,@view(theta[1:count]))
                copyto!(candidate,gx)
                mul!(candidate,@view(DG[:,1:count]),@view(theta[1:count]),-1.0,1.0)
                if all(isfinite,candidate)
                    apply!(work,candidate)
                    @inbounds @simd for i in 1:n;x[i]=work[i]-candidate[i];end
                    if fixed_norm(x)<=safeguard*q*fixed_norm(fx)*(1+1e-12)
                        x,candidate=candidate,x;gx,work=work,gx
                        accepted+=1;use_acceleration=true
                    end
                end
            end
            if !use_acceleration;rejected+=1;end
        end
        if !use_acceleration
            x,gx=gx,x;apply!(gx,x)
        end
    end
    (x=x,iterations=maxiter,residual=res,converged=false,accepted=accepted,rejected=rejected)
end
function sample_forest!(rng,g,roots,successor)
    fill!(roots,0);steps=0;n=length(roots)
    @inbounds for start in 1:n
        roots[start]!=0 && continue
        u=start
        while roots[u]==0
            steps+=1
            lo=g.ptr[u];hi=g.ptr[u+1]-1
            if g.unweighted
                choice=rand(rng,0:hi-lo+1)
                if choice==0;roots[u]=u;break;end
                successor[u]=g.nbr[lo+choice-1]
            else
                a=rand(rng)*(1+g.degree[u])
                if a<1;roots[u]=u;break;end
                idx=searchsortedfirst(@view(g.cumulative[lo:hi]),a-1)+lo-1
                successor[u]=g.nbr[idx]
            end
            u=successor[u]
        end
        root=roots[u];u=start
        while roots[u]==0;roots[u]=root;u=successor[u];end
    end
    steps
end
end
