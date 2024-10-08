# TransPed-Mutant: Automatically optimize mutated peptide program from the Transformer-derived self-attention model to predict peptide and HLA binding



**Parameter:**
- peptide_file: type = str, help = the filename of the .fasta file contains peptides
- HLA_file: type = str, help = the filename of the .fasta file contains sequence
- threshold: type = float, default = 0.5, help = the threshold to define predicted binder, float from 0 - 1, the recommended value is 0.5
- cut_peptide: type = bool, default = True, help = Whether to split peptides larger than cut_length?
- cut_length: type = int, default = 9, help = if there is a peptide sequence length > 15, we will segment the peptide according the length you choose, from 8 - 15
- output_dir: type = str, help = The directory where the output results are stored.
- output_attention, type = bool, default = True, help = Output the mutual influence of peptide and HLA on the binding?
- output_heatmap: type = bool, default = True, help = Visualize the mutual influence of peptide and HLA on the binding?
- output_mutation: type = bool, default = True, help = Whether to perform mutations with better affinity for each sample?
