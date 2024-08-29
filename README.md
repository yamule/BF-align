# BF-align (Brute Force aligner)

Fully sequence independent protein structure alignment with CUDA.

## Protocol

1. Construct frames for all of residues using backbone N, CA, C atoms or 3 CAs.
2. Perform all-against-all structural alignment with frames.
3. Optional: Re-align atoms using Biopython's SVDSuperimposer, using the result of step 2 as the initial 'seed'.

This tool is very slow with [CUDA](https://developer.nvidia.com/cuda-zone) and extremely slow without CUDA.
I recommend to install TM-align, MM-align, or US-align because it will be much easier than installing CUDA.

## How to use

`python scripts/bfalign.py --file1 example_files/sample_input_T1065.pdb --file2 example_files/7m5f.pdb --device cuda --realign True`

## Dependency

Biopython: https://github.com/biopython/biopython
pytorch: https://github.com/pytorch/pytorch

## License

MIT License

## Disclaimer

English translations are mainly done by the Claude 3.5 Sonnet AI.
