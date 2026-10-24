function query_certificate(x,r;metric=:gamma,delta=nothing,delta_max=nothing,
                           symmetric=false,scale=nothing,bound=:runtime)
    metric in (:gamma,:v) || throw(ArgumentError("unknown metric"))
    bound in (:runtime,:norm) || throw(ArgumentError("unknown certificate"))
    n=length(x)
    length(r)==n || throw(DimensionMismatch("residual"))
    isnothing(scale) || length(scale)==n || throw(DimensionMismatch("residual scale"))
    metric==:v && (isnothing(delta) || length(delta)!=n) &&
        throw(DimensionMismatch("variance weights"))
    q=0.0; ax=0.0; mx=0.0; weights=0.0; actual_max=0.0
    ri=0.0; rone=0.0; rtwo=0.0
    @inbounds for i in eachindex(x,r)
        a=abs(x[i]);w=metric==:gamma ? 1.0 : delta[i]
        w>=0 || throw(ArgumentError("nonnegative variance weights required"))
        actual_max=max(actual_max,w)
        z=abs(isnothing(scale) ? r[i] : r[i]/scale[i])
        q+=w*a*a;ax+=w*a;mx=max(mx,w*a);weights+=w
        ri=max(ri,z);rone+=z;rtwo+=z*z
    end
    all(isfinite,(q,ax,mx,weights,ri,rone,rtwo)) ||
        return (value=q,certificate=Inf,norm_certificate=Inf)
    dm=isnothing(delta_max) || metric==:gamma ? actual_max : max(actual_max,delta_max)
    isfinite(dm) || return (value=q,certificate=Inf,norm_certificate=Inf)
    er=symmetric ? sqrt(rtwo) : metric==:gamma ? sqrt(n)*ri : rone
    old=2sqrt(q*dm)*er+dm*er^2
    bound==:norm && return (value=q,certificate=old,norm_certificate=old)
    cert=old
    if metric==:gamma || symmetric
        cert=min(cert,2ax*ri+weights*ri^2)
        if symmetric
            cert=min(cert,ri*(ax+dm))
        end
    end
    if metric==:v
        cert=min(cert,2mx*rone+dm*rone^2,rone*(mx+dm))
    end
    (value=q,certificate=cert,norm_certificate=old)
end
