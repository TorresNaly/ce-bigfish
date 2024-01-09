# C. elegans smFISH quantification pipeline
Repository to automatically process C. elegans smFISH images. This repository uses PySMB to  transfer data between Network-attached storage (NAS) and a remote or local server. Then it uses Cellpose to detect and segment embryos on microscope images. Big-FISH is used to quantify the number of spots per cell. Data is processed using Pandas.
