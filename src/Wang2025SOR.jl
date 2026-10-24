module Wang2025SOR

using LinearAlgebra, SparseArrays
using ..NoisyFJ: FJGraph, query_certificate
export SORWorkspace, blisor!, sor_gamma!

# Wang et al. (2025), Algorithm 3 and the Section 5.1.3 shift.
# https://arxiv.org/html/2507.14864v1
mutable struct SORWorkspace
    graph::FJGraph
    x::Vector{Float64}
    r::Vector{Float64}
    rhs::Vector{Float64}
    threshold::Vector{Float64}
    queue::Vector{Int}
    queued::BitVector
    head::Int
    count::Int
end

function SORWorkspace(g::FJGraph)
    n = length(g.c)
    n > 0 || throw(ArgumentError("empty graph"))
    SORWorkspace(g, zeros(n), zeros(n), zeros(n), zeros(n),
                 zeros(Int, n), falses(n), 1, 0)
end

@inline function enqueue!(w, v)
    if !w.queued[v]
        k = w.head + w.count
        n = length(w.queue)
        k > n && (k -= n)
        w.queue[k] = v
        w.queued[v] = true
        w.count += 1
    end
end

function reset_queue!(w; all_nodes=false)
    w.head = 1
    w.count = 0
    fill!(w.queued, false)
    @inbounds for v in eachindex(w.r)
        (all_nodes || abs(w.r[v]) > w.threshold[v]) && enqueue!(w, v)
    end
end

function initialize!(w, shift, relative_epsilon)
    # Store the centered answer: x = z_shifted - shift, since B*1 = 1.
    @inbounds for i in eachindex(w.x)
        w.x[i] = -shift
        w.r[i] = w.rhs[i] + shift
        w.threshold[i] = relative_epsilon * (w.rhs[i] + shift)
    end
    reset_queue!(w; all_nodes=true)
end

@inline function push_node!(w, omega)
    v = w.queue[w.head]
    w.head = w.head == length(w.queue) ? 1 : w.head + 1
    w.count -= 1
    w.queued[v] = false
    mass = w.r[v]
    step = omega * w.graph.c[v] * mass
    w.x[v] += step
    A = w.graph.A
    # Updating x[v] changes residuals at in-neighbors: r -= B[:,v]*step.
    @inbounds for p in nzrange(A, v)
        u = A.rowval[p]
        w.r[u] += A.nzval[p] * step
        abs(w.r[u]) > w.threshold[u] && enqueue!(w, u)
    end
    w.r[v] = (1 - omega) * mass
    abs(w.r[v]) > w.threshold[v] && enqueue!(w, v)
    A.colptr[v + 1] - A.colptr[v]
end

function refresh_residual!(w)
    mul!(w.r, w.graph.B, w.x)
    @inbounds @simd for i in eachindex(w.r)
        w.r[i] = w.rhs[i] - w.r[i]
    end
end

function validate_options(omega, epsilon, shift, sigma, maxpush, check_every)
    isfinite(omega) && 0 < omega < 2 || throw(ArgumentError("omega must be in (0,2)"))
    isfinite(epsilon) && 0 < epsilon < 1 || throw(ArgumentError("relative epsilon must be in (0,1)"))
    isfinite(shift) && shift >= 0 || throw(ArgumentError("nonnegative finite shift required"))
    isfinite(sigma) && sigma > 0 || throw(ArgumentError("positive finite sigma required"))
    maxpush >= 0 && check_every > 0 || throw(ArgumentError("invalid work limit/check interval"))
end

function run!(w; omega, maxpush, check_every, gamma_tolerance=nothing)
    pushes = 0
    edge_updates = 0
    checks = 0
    refinements = 0
    next_check = check_every
    while true
        while w.count > 0 && pushes < maxpush && pushes < next_check
            edge_updates += push_node!(w, omega)
            pushes += 1
        end
        checks += 1
        candidate = query_certificate(w.x, w.r)
        finite = isfinite(candidate.certificate)
        verify = w.count == 0 || pushes >= maxpush || !finite ||
                 (!isnothing(gamma_tolerance) && candidate.certificate <= gamma_tolerance)
        if verify
            refresh_residual!(w)
            candidate = query_certificate(w.x, w.r)
            finite = isfinite(candidate.certificate)
            accepted = finite && (isnothing(gamma_tolerance) ?
                all(i -> abs(w.r[i]) <= w.threshold[i], eachindex(w.r)) :
                candidate.certificate <= gamma_tolerance)
            reason = accepted ? :converged : !finite ? :nonfinite : :maxpush
            if accepted || !finite || pushes >= maxpush
                return (x=w.x, value=candidate.value, certificate=candidate.certificate,
                        residual=norm(w.r, Inf), converged=accepted, reason=reason,
                        pushes=pushes, edge_updates=edge_updates, checks=checks,
                        refinements=refinements, omega=omega)
            end
            reset_queue!(w)
            if w.count == 0
                @inbounds for i in eachindex(w.threshold)
                    w.threshold[i] *= 0.1
                end
                refinements += 1
                reset_queue!(w)
            end
        end
        next_check = pushes + check_every
    end
end

"""ImprovedBLISOR with the paper's shifted componentwise relative stopping rule."""
function blisor!(w::SORWorkspace, b; epsilon=0.01, omega=1.2,
                 shift=1e-5, sigma=1e-5, maxpush=100_000_000,
                 check_every=length(w.x))
    validate_options(omega, epsilon, shift, sigma, maxpush, check_every)
    length(b) == length(w.x) || throw(DimensionMismatch("right-hand side"))
    all(v -> isfinite(v) && v >= 0, b) || throw(ArgumentError("nonnegative finite opinions required"))
    copyto!(w.rhs, b)
    initialize!(w, shift, epsilon * sigma / (shift + sigma))
    run!(w; omega, maxpush, check_every)
end

"""BLISOR updates for gamma_j, stopped by the shared absolute output certificate."""
function sor_gamma!(w::SORWorkspace, j; epsilon=1e-8, omega=1.2,
                    shift=1e-5, sigma=1e-5, relative_epsilon=nothing,
                    maxpush=100_000_000, check_every=length(w.x))
    validate_options(omega, something(relative_epsilon,0.01), shift, sigma, maxpush, check_every)
    isfinite(epsilon) && epsilon > 0 || throw(ArgumentError("positive finite output tolerance required"))
    1 <= j <= length(w.x) || throw(BoundsError(w.rhs, j))
    fill!(w.rhs, 0)
    w.rhs[j] = 1
    # For a unit query, ||r||∞ <= rho implies |x_i| <= 1+rho.
    # rho=min(epsilon,1)/(4n) therefore suffices for the output certificate.
    local_epsilon = isnothing(relative_epsilon) ?
        min(epsilon,1) / (4length(w.x)*(1+shift)) :
        relative_epsilon * sigma / (shift + sigma)
    initialize!(w, shift, local_epsilon)
    run!(w; omega, maxpush, check_every, gamma_tolerance=epsilon)
end

end
