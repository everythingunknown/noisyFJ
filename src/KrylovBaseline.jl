module KrylovBaseline
using LinearAlgebra
export KrylovWorkspace, bicgstab!, gmres!

struct KrylovWorkspace
    x::Vector{Float64}; r::Vector{Float64}; shadow::Vector{Float64}
    p::Vector{Float64};v::Vector{Float64};s::Vector{Float64};t::Vector{Float64}
    phat::Vector{Float64};shat::Vector{Float64};trial::Vector{Float64}
    V::Matrix{Float64};H::Matrix{Float64};cs::Vector{Float64};sn::Vector{Float64};z::Vector{Float64}
end
function KrylovWorkspace(n;restart=30)
    vecs=ntuple(_->zeros(n),10)
    KrylovWorkspace(vecs...,zeros(n,restart+1),zeros(restart+1,restart),zeros(restart),zeros(restart),zeros(restart+1))
end

function check!(g,b,x,scratch,stop)
    mul!(scratch,g.B,x)
    @. scratch=b-scratch
    stop(x,scratch)
end

function bicgstab!(g,b,w;stop,maxiter=20000)
    x,r,shadow,p,v,s,t,phat,shat,trial=w.x,w.r,w.shadow,w.p,w.v,w.s,w.t,w.phat,w.shat,w.trial
    fill!(x,0);copyto!(r,b);copyto!(shadow,r);fill!(p,0);fill!(v,0)
    alpha=1.0;omega=1.0;previous=1.0;matvecs=0
    for it in 1:maxiter
        rho=dot(shadow,r)
        if !isfinite(rho) || abs(rho)<=floatmin(Float64)
            return (x=x,iterations=it-1,matvecs,converged=false,status="rho_breakdown")
        end
        beta=(rho/previous)*(alpha/omega)
        @. p=r+beta*(p-omega*v)
        @. phat=g.c*p
        mul!(v,g.B,phat);matvecs+=1
        denominator=dot(shadow,v)
        if !isfinite(denominator) || abs(denominator)<=floatmin(Float64)
            return (x=x,iterations=it-1,matvecs,converged=false,status="alpha_breakdown")
        end
        alpha=rho/denominator
        @. s=r-alpha*v
        @. trial=x+alpha*phat
        if stop(trial,s)
            matvecs+=1
            if check!(g,b,trial,t,stop)
                copyto!(x,trial)
                return (x=x,iterations=it,matvecs,converged=true,status="converged")
            end
        end
        @. shat=g.c*s
        mul!(t,g.B,shat);matvecs+=1
        denominator=dot(t,t)
        if !isfinite(denominator) || denominator<=floatmin(Float64)
            return (x=x,iterations=it,matvecs,converged=false,status="omega_breakdown")
        end
        omega=dot(t,s)/denominator
        if !isfinite(omega) || abs(omega)<=floatmin(Float64)
            return (x=x,iterations=it,matvecs,converged=false,status="omega_breakdown")
        end
        @. x=trial+omega*shat
        @. r=s-omega*t
        if stop(x,r)
            matvecs+=1
            if check!(g,b,x,t,stop)
                return (x=x,iterations=it,matvecs,converged=true,status="converged")
            end
        end
        previous=rho
    end
    (x=x,iterations=maxiter,matvecs,converged=false,status="iteration_limit")
end

function gmres!(g,b,w;stop,restart=30,maxiter=20000)
    x,r,work,trial=w.x,w.r,w.p,w.trial
    V,H,cs,sn,z=w.V,w.H,w.cs,w.sn,w.z
    n=length(b);restart=min(restart,n,size(H,2));fill!(x,0)
    iterations=0;matvecs=0
    while iterations<maxiter
        mul!(r,g.B,x);matvecs+=1
        @. r=g.c*(b-r)
        beta=norm(r)
        if beta==0
            return (x=x,iterations,matvecs,converged=true,status="converged")
        end
        @views V[:,1].=r./beta
        fill!(H,0);fill!(z,0);z[1]=beta
        for k in 1:min(restart,maxiter-iterations)
            mul!(work,g.B,@view(V[:,k]));matvecs+=1
            @. work=g.c*work
            for j in 1:k
                h=dot(@view(V[:,j]),work);H[j,k]=h
                @views @. work-=h*V[:,j]
            end
            H[k+1,k]=norm(work)
            happy=H[k+1,k]<=eps(Float64)*max(1.0,norm(@view(H[1:k,k])))
            if !happy; @views V[:,k+1].=work./H[k+1,k]; end
            for j in 1:k-1
                a=H[j,k];bb=H[j+1,k]
                H[j,k]=cs[j]*a+sn[j]*bb;H[j+1,k]=-sn[j]*a+cs[j]*bb
            end
            h=hypot(H[k,k],H[k+1,k])
            if h==0;return (x=x,iterations,matvecs,converged=false,status="arnoldi_breakdown");end
            cs[k]=H[k,k]/h;sn[k]=H[k+1,k]/h
            H[k,k]=h;H[k+1,k]=0
            z[k+1]=-sn[k]*z[k];z[k]=cs[k]*z[k]
            y=UpperTriangular(@view(H[1:k,1:k]))\@view(z[1:k])
            copyto!(trial,x);mul!(trial,@view(V[:,1:k]),y,1.0,1.0)
            iterations+=1
            matvecs+=1
            if check!(g,b,trial,r,stop)
                copyto!(x,trial)
                return (x=x,iterations,matvecs,converged=true,status="converged")
            end
            if happy || k==restart
                copyto!(x,trial);break
            end
        end
    end
    (x=x,iterations,matvecs,converged=false,status="iteration_limit")
end
end
