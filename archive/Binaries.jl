module Binaries
    using FastaIO
    using SHA

    include("CommonUtils.jl")

    figtree_binary  = joinpath(@__DIR__,"..","binaries","figtree-custom.jar")
    function getfigtreesvg(newick_file::AbstractString, width::Int=320, height::Int=500)
        return read(`java -jar $(figtree_binary) -graphic SVG -width $(width) -height $(height) $(newick_file)`, String)
    end

    function fasttreegtr(alignmentfile::AbstractString)
        fastastring = open(alignmentfile) do file
            read(file, String)
        end
        cachepath = joinpath(@__DIR__,"..","cache")
        mkpath(cachepath)
        cachefile = joinpath(cachepath, string(CommonUtils.sha256base36(fastastring), ".fasttreegtr.nwk"))
        if isfile(cachefile)
            newickstring = open(cachefile) do file
                read(file, String)
            end
            newickstring, cachefile
        else
            fasttreepath = fasttreepath = joinpath(@__DIR__,"..","binaries","FastTreeMP")
            if Sys.iswindows()
              fasttreepath = joinpath(@__DIR__,"..","binaries","FastTree.exe")
            end
            newickstring = read(`$fasttreepath -nt -gtr -nosupport $alignmentfile`, String)

            fout = open(cachefile, "w")
            print(fout, strip(newickstring))
            close(fout)
            return newickstring, cachefile
        end
    end

    function fasttreeaa(alignmentfile::AbstractString; branchsupport=false)
        fastastring = open(alignmentfile) do file
            read(file, String)
        end
        cachepath = joinpath(@__DIR__,"..","cache")
        mkpath(cachepath)
        cachepath = abspath(cachepath)
        println(cachepath)
        cachefile = joinpath(cachepath, string(CommonUtils.sha256base36(fastastring), ".fasttreeaa.nwk"))
        if isfile(cachefile)
            newickstring = open(cachefile) do file
                strip(read(file, String))
            end
            newickstring, cachefile
        else
            fasttreepath = joinpath(@__DIR__,"..","binaries","FastTreeMP")
            if Sys.iswindows()
              fasttreepath = abspath(joinpath(@__DIR__,"..","binaries","FastTree.exe"))
            end
            newickstring = ""
            if branchsupport
                newickstring = read(`$fasttreepath $alignmentfile`, String)
            else
                newickstring = read(`$fasttreepath -nosupport $alignmentfile`, String)
            end

            fout = open(cachefile, "w")
            print(fout, strip(newickstring))
            close(fout)
            return newickstring, cachefile
        end
    end

    #=
    function mafft(alignmentfile::AbstractString)
        fastastring = open(alignmentfile) do file
            read(file, String)
        end
        cachepath = joinpath(@__DIR__,"..","cache")
        mkpath(cachepath)
        cachefile = joinpath(cachepath, string(CommonUtils.sha256base36(fastastring), ".mafft.fas"))
        if isfile(cachefile) && 1 == 2
            mafft_alignment = open(cachefile) do file
                read(file, String)
            end
            mafft_alignment, cachefile
        else
            if Sys.iswindows()
              mafft_windows = abspath(joinpath(@__DIR__,"..","binaries","mafft-7.402-win64-signed","mafft-win","mafft.bat"))

              cmd = Cmd(`$mafft_windows $(basename(alignmentfile))`, windows_verbatim=true, dir=dirname(alignmentfile))
              run(cmd)
              mafft_alignment = read(cmd, String)
              println("B")
              fout = open(cachefile, "w")
              print(fout, strip(mafft_alignment))
              close(fout)
              return mafft_alignment, cachefile
            else
              return "",""
            end
        end
    end
    =#

    function muscle(alignmentfile::AbstractString)
        fastastring = open(alignmentfile) do file
            read(file, String)
        end
        cachepath = joinpath(@__DIR__,"..","cache")
        mkpath(cachepath)
        cachepath = abspath(cachepath)
        cachefile = joinpath(cachepath, string(CommonUtils.sha256base36(fastastring), ".muscle.fas"))
        if isfile(cachefile)
            muscle_alignment = open(cachefile) do file
                strip(read(file, String))
            end
            return muscle_alignment, cachefile
        else
            if Sys.iswindows()
              muscle_windows = joinpath(@__DIR__,"..","binaries","muscle3.8.31_i86win32.exe")
              read(`$muscle_windows -in $alignmentfile -out $cachefile`, String)
              muscle_alignment = open(cachefile) do file
                read(file, String)
              end
              return muscle_alignment, cachefile
            elseif Sys.islinux()
              muscle_windows = joinpath(@__DIR__,"..","binaries","muscle3.8.31_i86linux64")
              read(`$muscle_windows -in $alignmentfile -out $cachefile`, String)
              muscle_alignment = open(cachefile) do file
                read(file, String)
              end
              return muscle_alignment, cachefile
            else
              return "",""
            end
        end
    end
end
