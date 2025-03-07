## Installation

First install the R development tools. On MacOS, install Xcode command line tools
by running `xcode-select --install`. On Windows, install
[Rtools](https://cran.r-project.org/bin/windows/Rtools/).

Then install the `remotes` package:

```r
install.packages("remotes")
```

Now install the Rust compiler and toolchain as described [here](gourd/INSTALL).

Finally, install the package itself:

```
remotes::install_github("dbdahl/gourd-package/gourd")
```

## Usage

The main functions of interest are `ShrinkagePartition`, `CRPPartition`, `samplePartition`, and `prPartition`.
Please see the documentation and examples in the R package. 


