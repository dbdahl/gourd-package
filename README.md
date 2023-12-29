## Installation

You can install the latest version from source.

First install the development tools. On MacOS, install Xcode command line tools
by running `sudo xcode-select --install`. On Windows, install
[Rtools](https://cran.r-project.org/bin/windows/Rtools/).

Then install the `remotes` and `cargo` packages:

```r
install.packages(c("cargo", "remotes"))
```

Now install the Rust toolchain:

```
cargo::install(force = TRUE)
```

Finally, install the package itself:

```
remotes::install_github("dbdahl/gourd-package/gourd")
```

## Usage

The main functions of interest are `ShrinkagePartition`, `CRPPartition`, `samplePartition`, and `prPartition`.
Please see the documentation and examples in the R package. 


