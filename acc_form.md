<!--HOW TO COMPLETE THIS FORM:-->
<!--
1. Checkboxes in this document appear as follows: 

- [ ] This is a checkbox 

To check a checkbox, replace [ ] by [x], as follows: 

- [x] This is a checked checkbox 

Note that current versions of RStudio for Mac (this will change with RStudio versions 1.3 and higher) will not create a formatted checkbox but will leave the original characters, i.e., literally "[ ]" or "[x]". It's fine to submit a PDF in this form.
 
2. For text answers, simply type the relevant text in the areas indicated. A blank line starts a new paragraph. 
 
3. Comments (like these instructions) provide additional instructions throughout the form. There is no need to remove them; they will not appear in the compiled document. 

4. If you are comfortable with Markdown syntax, you may choose to include any Markdown-compliant formatting in the form. For example, you may wish to include R code chunks and compile this document in R Markdown.
-->

This form documents the artifacts associated with the article (i.e., the
data and code supporting the computational findings) and describes how
to reproduce the findings.

Part 1: Data
============

-   [ ] This paper does not involve analysis of external data (i.e., no
    data are used or the only data are generated by the authors via
    simulation in their code).

<!--
If box above is checked and if no simulated/synthetic data files are provided by the authors, please skip directly to the Code section. Otherwise, continue.
-->

-   [x] I certify that the author(s) of the manuscript have legitimate
    access to and permission to use the data used in this manuscript.

<!-- If data are simulated using random number generation, please be sure to set the random number seed in the code you provide -->

Abstract
--------

<!--
Provide a short (< 100 words), high-level description of the data
-->

Availability
------------

-   [x] Data **are** publicly available.
-   [ ] Data **cannot be made** publicly available.

If the data are publicly available, see the *Publicly available data*
section. Otherwise, see the *Non-publicly available data* section,
below.

### Publicly available data

-   [x] Data are available online at: Various locations. See the
    Appendix of the NIMIWAE manuscript (link to paper here). For your
    convenience, raw UCI and Physionet data can be found here: (link to
    raw\_data folder here). Missingness is simulated on top of the UCI
    data. The missingness-simulated datasets can be reproduced by using
    the code here (using the default seed as provided), but these
    datasets can also be provided by request.

-   [ ] Data are available as part of the paper’s supplementary
    material.

-   [ ] Data are publicly available by request, following the process
    described here:

-   [ ] Data are or will be made available through some other mechanism,
    described here:

Description
-----------

### File format(s)

<!--
Check all that apply
-->

-   [x] CSV or other plain text.
-   [x] Software-specific binary format (.Rda, Python pickle, etc.):
    pkcle
-   [ ] Standardized binary format (e.g., netCDF, HDF5, etc.):
-   [ ] Other (please specify):

### Data dictionary

<!--
A data dictionary provides information that allows users to understand the meaning, format, and use of the data.
-->

-   [x] Provided by authors in the following file(s): See the Appendix
    of the NIMIWAE manuscript

-   [ ] Data file(s) is(are) self-describing (e.g., netCDF files)

-   [ ] Available at the following URL:

### Additional Information (optional)

<!-- 
OPTIONAL: Provide any additional details that would be helpful in understanding the data. If relevant, please provide unique identifier/DOI/version information and/or license/terms of use.
-->

Part 2: Code
============

Abstract
--------

<!--
Provide a short (< 100 words), high-level description of the code. If necessary, more details can be provided in files that accompany the code.
-->

Description
-----------

### Code format(s)

<!--
Check all that apply
-->

-   [x] Script files
    -   [x] R
    -   [x] Python
    -   [ ] Matlab
    -   [ ] Other:
-   [x] Package
    -   [x] R
    -   [x] Python
    -   [ ] MATLAB toolbox
    -   [ ] Other:
-   [ ] Reproducible report
    -   [ ] R Markdown
    -   [ ] Jupyter notebook
    -   [ ] Other:
-   [ ] Shell script
-   [ ] Other (please specify):

### Supporting software requirements

#### Version of primary software used

<!--
(e.g., R version 3.6.0)
-->

-   R version 3.6.1
-   Python version 3.6.3

#### Libraries and dependencies used by the code

<!--
Include version numbers (e.g., version numbers for any R or Python packages used)
-->

R packages:

-   reticulate (1.13)
-   NIMIWAE (0.1.0)

Python modules:

-   numpy (1.18.1)
-   pandas (1.5.0)
-   scipy (1.4.1)
-   torch (1.5.0)
-   tensorflow (2.2.0)
-   sklearn (0.22.1)
-   argparse (1.1)
-   tqdm (4.42.1)

### Supporting system/hardware requirements (optional)

<!--
OPTIONAL: System/hardware requirements including operating system with version number, access to cluster, GPUs, etc.
-->

This code requires access to a cuda-enabled GPU.

### Parallelization used

-   [x] No parallel code used
-   [ ] Multi-core parallelization on a single machine/node
    -   Number of cores used:
-   [ ] Multi-machine/multi-node parallelization
    -   Number of nodes and cores used: 3 nodes, 243 cores

### License

-   [x] MIT License (default)
-   [ ] BSD
-   [ ] GPL v3.0
-   [ ] Creative Commons
-   [ ] Other: (please specify below)

### Additional information (optional)

<!--
OPTIONAL: By default, submitted code will be published on the JASA GitHub repository (http://github.com/JASA-ACS) as well as in the supplementary material. Authors are encouraged to also make their code available in a public repository. If relevant, please provide unique identifier/DOI/version information.

# Part 3: Reproducibility workflow

<!--
The materials provided should provide a straightforward way for reviewers and readers to reproduce analyses with as few steps as possible. 
-->

Scope
-----

The provided workflow reproduces:

-   [x] Any numbers provided in text in the paper
-   [x] All tables and figures in the paper
-   [ ] Selected tables and figures in the paper, as explained and
    justified below:

Workflow
--------

Run `runComparisons.R` script to train all models (time-consuming).
Then, run `SummarizeResults.R` script to reproduce Figures 2 and 3, and
the results in Table 1.

### Format(s)

<!--
Check all that apply
-->

-   [ ] Single master code file
-   [ ] Wrapper (shell) script(s)
-   [ ] Self-contained R Markdown file, Jupyter notebook, or other
    literate programming approach
-   [ ] Text file (e.g., a readme-style file) that documents workflow
-   [ ] Makefile
-   [ ] Other (more detail in *Instructions* below)

### Instructions

<!--
Describe how to use the materials provided to reproduce analyses in the manuscript. Additional details can be provided in file(s) accompanying the reproducibility materials.
-->

### Expected run-time

Approximate time needed to reproduce the analyses on a standard desktop
machine:

-   [ ] \< 1 minute
-   [ ] 1-10 minutes
-   [ ] 10-60 minutes
-   [ ] 1-8 hours
-   [x] \> 8 hours
-   [ ] Not feasible to run on a desktop machine, as described here

Training the deep learning architectures can be very time-consuming.
Summarizing the results should not take any longer than 10 minutes.

### Additional information (optional)

<!--
OPTIONAL: Additional documentation provided (e.g., R package vignettes, demos or other examples) that show how to use the provided code/software in other settings.
-->

Notes (optional)
================

<!--
OPTIONAL: Any other relevant information not covered on this form. If reproducibility materials are not publicly available at the time of submission, please provide information here on how the reviewers can view the materials.
-->