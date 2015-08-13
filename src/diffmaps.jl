# Diffusion maps
# --------------
# Diffusion maps,
# Coifman, R. & Lafon, S., Applied and Computational Harmonic Analysis, Elsevier, 2006, 21, 5-30

#### DiffMap type
immutable DiffMap <: SpectralResult
    t::Int
    ε::Float64
    λ::Vector{Float64}
    K::Matrix{Float64}
    proj::Projection

    DiffMap(t::Int, ε::Float64, λ::Vector{Float64}, K::Matrix{Float64}, proj::Projection) = new(t, ε, λ, K, proj)
end

## properties
outdim(M::DiffMap) = size(M.proj, 1)
projection(M::DiffMap) = M.proj
kernel(M::DiffMap) = M.K

## show & dump
function show(io::IO, M::DiffMap)
    print(io, "Diffusion Maps(outdim = $(outdim(M)), t = $(M.t), ε = $(M.ε))")
end

function dump(io::IO, M::DiffMap)
    show(io, M)
    println(io, "kernel: ")
    Base.showarray(io, M.K, header=false, repr=false)
    println(io)
    println(io, "projection:")
    Base.showarray(io, projection(M), header=false, repr=false)
    println(io, "eigenvalues:")
    Base.showarray(io,M.λ,header=false,repr=false)
end

## interface functions
function transform(::Type{DiffMap}, X::DenseMatrix{Float64}; d::Int=2, α::Int=1, ε::Float64=1.0)

    #    transform!(fit(UnitRangeTransform, X), X)
    X = (X-minimum(X))./(maximum(X)-minimum(X))
    # Replace this when UnitRangeTransform becomes more widely available

    K = exp(-pairwise(SqEuclidean(),X)./ε)
    p = sum(K, 1)'
    K ./= ((p * p') .^ α)
    p = sqrt(sum(K, 1))'
    K ./= (p * p')

    U, s, V = tsvd(K, d+1)
    U ./= U[:,1]
    Y = U[:,2:end]

    return DiffMap(α, ε, s[2:end], K, Y')
end

# Nystrom extension for regression taken from Freeman et al. MNRAS 2009
function nystrom(dm::DiffMap, X::DenseMatrix{Float64}, Y::DenseMatrix{Float64})
    W = exp(-pairwise(SqEuclidean(),X,Y)./dm.ε) #Compute the kernel
    W./=sum(W,1) # Row-normalize except our rows are columns
    dm.proj*W #Eq. 4 from Freeman but reversed because everything is transposed
    # and we don't multiply by the eigenvalues in the projection matrix
end
