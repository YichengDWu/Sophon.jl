using Sophon
using Documenter
using DocumenterCitations
using DocThemeIndigo

indigo = DocThemeIndigo.install(Sophon)

bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"); sorting=:nyt)

DocMeta.setdocmeta!(Sophon, :DocTestSetup, :(using Sophon); recursive=true)

makedocs(bib; modules=[Sophon],
         repo="https://github.com/YichengDWu/Sophon.jl/blob/{commit}{path}#{line}",
         sitename="Sophon.jl",
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true",
                                canonical="https://YichengDWu.github.io/Sophon.jl",
                                edit_link="main", assets=String[indigo]),
         strict=[
             :doctest,
             :linkcheck,
             :parse_error,
             :example_block,
             # Other available options are
             # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
         ], authors="Yicheng Wu",
         pages=[
             "Home" => "index.md",
             "Tutorials" => [
                 "Fitting a nonlinear discontinuous function" => "tutorials/discontinuous.md",
                 "1D Poisson's Equation" => "tutorials/poisson.md",
                 "1D Convection Equation" => "tutorials/convection.md",
                 "2D Helmholtz Equation" => "tutorials/helmholtz.md",
                 "Allen-Cahn equation with Sequential Training" => "tutorials/allen_cahn.md",
                 #"1D Wave Equation" => "tutorials/wave.md",
             ],
             "References" => "references.md",
         ])

deploydocs(; repo="github.com/YichengDWu/Sophon.jl", devbranch="main")
