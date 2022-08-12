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
         strict=[
             :doctest,
             :linkcheck,
             :parse_error,
             :example_block,
             # Other available options are
             # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
         ],
         pages=[
             "Home" => "index.md",
             "Tutorials" => [
                 "Fitting a nonlinear discontinuous function" => "tutorials/discontinuous.md",
                 "1D Poisson's Equation" => "tutorials/poisson.md",
             ],
         ])

deploydocs(; repo="github.com/MilkshakeForReal/Sophon.jl", devbranch="main")
