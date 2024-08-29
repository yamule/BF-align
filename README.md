# Brute Force aligner

Fully sequence independent protein structural alignment with CUDA.

## Protocol

1. Construct frames for all of residues using backbone N, CA, C atoms or 3 CAs.
2. Perform all-against-all structural alignment with frames.
3. Re-align atoms with Biopython's SVDSuperimposer using the result of step 2 as 'seed', if you needed.

This tool is super slow with CUDA & super super slow without CUDA.
I recommend to install TM-align, MM-align, or US-align because it will be much easier than installing CUDA.

## Dependency

Biopython

## License

MIT License

## Disclaimer

English translations are mainly done by Claude3.5 Sonnet.
