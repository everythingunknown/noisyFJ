module MixedReference
using ..NoisyFJ, LinearAlgebra
export refined_reference, dd_residual!

@inline function twosum(a,b)
    s=a+b;bb=s-a;s,(a-(s-bb))+(b-bb)
end
@inline function ddadd(ah,al,bh,bl)
    s,e=twosum(ah,bh);t,f=twosum(al,bl);e+=t
    h,l=twosum(s,e);twosum(h,l+f)
end
@inline function ddmul(a,h,l)
    p=a*h;e=fma(a,h,-p)+a*l;twosum(p,e)
end
function dd_residual!(rh,rl,g,xh,xl,j)
    @inbounds for i in eachindex(xh)
        a,b=ddmul(-(1+g.degree[i]),xh[i],xl[i])
        rh[i],rl[i]=i==j ? ddadd(a,b,1.0,0.0) : (a,b)
    end
    A=g.A
    @inbounds for k in eachindex(xh)
        (xh[k]==0 && xl[k]==0) && continue
        for z in A.colptr[k]:A.colptr[k+1]-1
            i=A.rowval[z]
            a,b=g.unweighted ? (xh[k],xl[k]) : ddmul(A.nzval[z],xh[k],xl[k])
            rh[i],rl[i]=ddadd(rh[i],rl[i],a,b)
        end
    end
    maximum(i->abs(rh[i])+abs(rl[i]),eachindex(rh))
end
function refined_reference(g,j;epsilon=1e-12,workspace=nothing)
    n=length(g.c);b=zeros(n);b[j]=1.0
    ws=isnothing(workspace) ? FJSolveWorkspace(g;history=3) : workspace
    initial=solve_fj(g,b;method=:anderson,tol=1e-9,maxiter=500,workspace=ws)
    xh=copy(initial.x);xl=zeros(n);rh=zeros(n);rl=zeros(n)
    steps=initial.iterations
    for refinement in 0:6
        eta=dd_residual!(rh,rl,g,xh,xl,j)
        ax=sum(i->abs(xh[i])+abs(xl[i]),eachindex(xh))
        guard=100eps(Float64)^2*(2+maximum(g.degree))^2*max(1.0,maximum(abs,xh))
        cert=2ax*(eta+guard)+n*(eta+guard)^2
        if cert<=epsilon
            value=setprecision(256) do
                sum(i->(BigFloat(xh[i])+BigFloat(xl[i]))^2,eachindex(xh))
            end
            return (value=value,certificate=cert,iterations=steps,converged=true,refinements=refinement)
        end
        @. b=rh+rl
        correction=solve_fj(g,b;method=:anderson,tol=max(eta*1e-6,1e-30),maxiter=1000,workspace=ws)
        steps+=correction.iterations
        @inbounds for i in eachindex(xh)
            xh[i],xl[i]=ddadd(xh[i],xl[i],correction.x[i],0.0)
        end
    end
    error("mixed-precision reference did not reach $epsilon at node $j")
end
end
