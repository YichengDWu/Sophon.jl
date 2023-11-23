using Sophon
using Documenter
using DocumenterCitations
using DocThemeIndigo

indigo = DocThemeIndigo.install(Sophon)

bib = CitationBibliography(joinpath(@__DIR__, "bibliography.bib"))

DocMeta.setdocmeta!(Sophon, :DocTestSetup, :(using Sophon); recursive=true)

makedocs(; plugins=[bib], modules=[Sophon], sitename="Sophon.jl",
         repo="https://github.com/YichengDWu/Sophon.jl/blob/{commit}{path}#{line}",
         format=Documenter.HTML(; prettyurls=get(ENV, "CI", "false") == "true",
                                canonical="https://YichengDWu.github.io/Sophon.jl",
                                edit_link="main", assets=String[indigo]),
         authors="Yicheng Wu",
         pages=[
             "Home" => "index.md",
             "Tutorials" => [
                 "Introduction with Lotka-Volterra System"=> "tutorials/ode.md",
                 "Fitting a nonlinear discontinuous function" => "tutorials/discontinuous.md",
                 "1D Multi-scale Poisson's Equation" => "tutorials/poisson.md",
                 "1D Convection Equation" => "tutorials/convection.md",
                 "2D Helmholtz Equation" => "tutorials/helmholtz.md",
                 "Allen-Cahn Equation with Sequential Training" => "tutorials/allen_cahn.md",
                 "Schrödinger Equation: A PDE System with Resampling" => "tutorials/SchrödingerEquation.md",
                 "Poisson equation over an L-shaped domain" => "tutorials/L_shape.md",
                 "Inverse problem for the wave equation with unknown velocity field" => "tutorials/waveinverse2.md",
                 "SOD Shock Tube Problem" => "tutorials/sod.md",
                 #"1D Wave Equation" => "tutorials/wave.md",
             ],
             "FAQ" => "qa.md",
             "API Reference" => "api.md",
             "References" => "references.md",
         ])

deploydocs(; repo="github.com/YichengDWu/Sophon.jl", devbranch="main", push_preview=true)
