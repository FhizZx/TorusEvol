

function TKF92TransitionMatrix(λ::T, μ::T, r::T, t::T) where T <: Real :: Matrix[T]
    transMat = Matrix{T}(0, NUM_ALIGN_STATES, NUM_ALIGN_STATES)
    β = β(λ, μ, t)
    # pxy = probability of transitioning from state x to state y
    # S = start, E = end
    # b = birth decision area, e = elongation decision area
    # f = fragment death decision area (match or del)
    # g = dead fragment # of children decision area (0 or >0)
    # i = insertion, d = deletion, m = match -> the actual emissions
    pSb = 1
    pbi = λ*β
    pbe = 1 - pbi
    pii = r + (1-r)*λ*β
    pie = 1 - pii
    pef = λ / μ
    peE = 1 - pef
    pfm = exp(-μ*t)
    pfd = 1 - pfm
    pmm = r
    pmb = 1 - pmm
    pdd = r
    pdg = 1-r
    pgi = (1 - μ * β - exp(-μ*t)) / (1-exp(-μ*t))
    pge = 1 - pgi

    transMat[START, INSERT] = pSb * pbi
    transMat[START, MATCH] = pSb * pbe * pef * pfm
    transMat[START, DELETE] = pSb * pbe * pef * pfd
    transMat[START, END] = pSb * pbe * peE

    transMat[INSERT, INSERT] = pii
    transMat[INSERT, MATCH] = pie * pef * pfm
    transMat[INSERT, DELETE] = pie * pef * pfd
    transMat[INSERT, END] = pie * peE

    transMat[MATCH, INSERT] = pmb * pbi
    transMat[MATCH, MATCH] = pmm + pmb * pbe * pef * pfm
    transMat[MATCH, DELETE] = pmb * pbe * pef * pfd
    transMat[MATCH, END] = pmb * pbe * peE

    transMat[DELETE, INSERT] = pdg * pgi
    transMat[DELETE, MATCH] = pdg * pge * pef * pfm
    transMat[DELETE, DELETE] = pdd + pdg * pge * pef * pfd
    transMat[DELETE, END] = pdg * pge * peE

    map!(x -> log(x), transMat, transMat)

    return transMat
end

β(λ, μ, t) = (1 - exp((λ - μ) * t)) / (μ - λ*exp((λ - μ) * t))
