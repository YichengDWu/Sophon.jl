using Sophon
using Documenter

DocMeta.setdocmeta!(Sophon, :DocTestSetup, :(using Sophon); recursive=true)

makedocs(; modules=[Sophon],
         authors="MilkshakeForReal <yicheng.wu@ucalgary.ca> and contributors",
         repo="https://github.com/MilkshakeForReal/Sophon.jl/blob/{commit}{path}#{line}",
         sitename="Sophon.jl",
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true",
                                canonical="https://MilkshakeForReal.github.io/Sophon.jl",
                                edit_link="main", assets=String[]),
         pages=["Home" => "index.md",
                "Tutorials" => [
                    "Fitting a nonlinear discontinuous function" => "tutorials/discontinuous.md",
                ]])

deploydocs(; repo="github.com/MilkshakeForReal/Sophon.jl", devbranch="main")
