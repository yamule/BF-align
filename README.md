# BF-align (Brute Force aligner)

Fully sequence independent protein structure alignment with pytorch, inspired by FAPE loss calculations by [AlphaFold2](https://github.com/google-deepmind/alphafold).

## Protocol

1. Construct frames for all of residues using backbone N, CA, C atoms or 3 CAs (although it may not be a fully independent).
2. Perform all-against-all structural alignment with frames.
3. Optional: Re-align atoms using Biopython's SVDSuperimposer, using the result of step 2 as the initial 'seed'.

This tool is very slow with [CUDA](https://developer.nvidia.com/cuda-zone) and extremely slow without CUDA.
I recommend to install [TM-align](https://zhanggroup.org/TM-align/), [MM-align](https://zhanggroup.org/MM-align/), or [US-align](https://github.com/pylelab/USalign) instead of this tool because it will be much easier than installing CUDA.

I am not familiar with pytorch and I am not good at programming. If you have ideas about speeding up the script, please improve it!

## Usage
```
usage: BF-align [-h] --file1 FILE1 --file2 FILE2 [--outfile OUTFILE] [--use_ca USE_CA] [--realign REALIGN]
                [--device DEVICE] [--chunk_size CHUNK_SIZE] [--max_realign MAX_REALIGN]

Fully sequence independent structure alignment.

optional arguments:
  -h, --help            show this help message and exit
  --file1 FILE1         Query protein structure in PDB format to be aligned to file2 structure.
  --file2 FILE2         Template protein structure in PDB format to align against.
  --outfile OUTFILE     Output file path of alignment result of file1 structure.
  --use_ca USE_CA       Use 3 CA atoms for alignment. 'true' or 'false'.
  --realign REALIGN     Perform re-alignment with Biopython's SVDSuperimposer. 'true' or 'false'.
  --device DEVICE       Computation device: 'cpu' or 'cuda'.
  --chunk_size CHUNK_SIZE
                        Chunk size when calculate TM-score with batch.
  --max_realign MAX_REALIGN
                        Maximum number of iteration to realign with SVDSuperimposer.
```

## Example

`python scripts/bfalign.py --file1 example_files/sample_input_T1065.pdb --file2 example_files/7m5f.pdb --device cuda --realign True`

## Dependency

Biopython: https://github.com/biopython/biopython

pytorch: https://github.com/pytorch/pytorch

## License

MIT License

## Disclaimer

English translations are mainly done by the Claude 3.5 Sonnet AI.
