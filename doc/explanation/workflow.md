# Workflow

```{graphviz}
digraph foo {
    LineCollection [shape=box]
    "Temperature bins" [shape=box]
    "Contribution functions" [shape=box]
    Density [shape=box]
    "Observed intensities" [shape=box]
    "Estimated DEM" [shape=box]

    "Calculate contribution functions"

    "Observed intensities" -> "Density"
    Density -> "Calculate contribution functions"
    "Calculate contribution functions" -> "Contribution functions"

    "Observed intensities" -> LineCollection
    "Contribution functions" -> LineCollection

    LineCollection -> "DEM inversion";
    "Temperature bins" -> "DEM inversion";

    "DEM inversion" -> "Estimated DEM"
}
```
