module SmallAllocationControls

export extra_counts, half_additions, extend_streams!, observation_errors

# Allocate a batch of extra measurements without removing previous observations.
# A heap is used only for the small-network adaptive control, where scores change
# between rounds. Static scores continue to use the original threshold allocator.
function extra_counts(scores, current, extra)
    n = length(scores)
    @assert length(current) == n && all(current .>= 1) && extra >= 0
    @assert all(x -> isfinite(x) && x >= 0, scores)
    counts = Int.(current)
    gain(i) = scores[i] / (Float64(counts[i]) * (counts[i] + 1))
    better(i, j) = gain(i) > gain(j) || (gain(i) == gain(j) && i < j)
    heap = collect(1:n)
    function down!(p)
        while 2p <= n
            q = 2p
            q < n && better(heap[q+1], heap[q]) && (q += 1)
            better(heap[q], heap[p]) || break
            heap[q], heap[p] = heap[p], heap[q]
            p = q
        end
    end
    for p in (n ÷ 2):-1:1
        down!(p)
    end
    for _ in 1:extra
        counts[heap[1]] += 1
        down!(1)
    end
    @assert sum(counts) == sum(current) + extra && all(counts .>= current)
    counts
end

function half_additions(scores)
    n = length(scores)
    order = sortperm(1:n; by=i -> (-scores[i], i))
    additions = zeros(Int, n)
    additions[order[1:n÷2]] .= 2
    isodd(n) && (additions[order[n÷2+1]] = 1)
    @assert sum(additions) == n
    additions
end

# Each node has one shared, lazily extended sequence of observation errors.
# Policies access only their own prefix, even if another policy requested more.
function extend_streams!(streams, amplitude, requested, rng)
    for i in eachindex(streams)
        while length(streams[i]) < requested[i]
            push!(streams[i], amplitude[i] * (2rand(rng) - 1))
        end
    end
end

function observation_errors(streams, counts)
    [sum(@view streams[i][1:counts[i]]) / counts[i] for i in eachindex(counts)]
end

end
