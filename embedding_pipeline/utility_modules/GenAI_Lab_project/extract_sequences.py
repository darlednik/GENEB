from pyfaidx import Fasta
import csv
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from tqdm import tqdm

def extract_sequences():
    def extract_sequences_from_eccDNA_Atlas():
      # Load reference genomes
      hg19_genome = Fasta('data/genomes/hg19.fa')
      galGal4_genome = Fasta('data/genomes/galGal4.fa')
      mm10_genome = Fasta('data/genomes/mm10.fa')
    
      # Load reference sequences
      Homo_sapiens_input = "data/raw/eccDNA_Atlas/Homo_sapiens.tsv"
      Gallus_gallus_input = "data/raw/eccDNA_Atlas/Gallus_gallus.tsv"
      Mus_musculus_input = "data/raw/eccDNA_Atlas/Mus_musculus.tsv"
    
      # Save sequences
      os.makedirs("data/preprocessed/eccDNA_Atlas", exist_ok=True)
    
      Homo_sapiens_output = "data/preprocessed/eccDNA_Atlas/Homo_sapiens.txt"
      Gallus_gallus_output = "data/preprocessed/eccDNA_Atlas/Gallus_gallus.txt"
      Mus_musculus_output = "data/preprocessed/eccDNA_Atlas/Mus_musculus.txt"
    
      # Function to extract sequences and write to a new file
      def extract_sequences(input_file, output_file, genome):
          with open(input_file, newline='', encoding='utf-8') as infile, \
              open(output_file, "w", newline='', encoding='utf-8') as outfile:
    
              reader = csv.DictReader(infile, delimiter='\t')
    
              for row in tqdm(reader, desc=f"Extracting {input_file}"):
                  try:
                      chrom = row['Chr']  # Extract chromosome name
                      eccDNA_type = row['eccDNA type']
                      start = int(float(row['Start'])) - 1  # Convert start position to integer (0-indexed)
                      end = int(float(row['End']))  # Convert end position to integer
                      seq = genome[chrom][start:end].seq.upper()  # Extract sequence from genome file
                      if eccDNA_type == "ecDNA" or 'N' in seq or end - start > 10000:  # Skip ecDNA and sequences with 'N' (unknown bases) and sequences too long
                          continue
                  except Exception as e:
                      continue
    
                  outfile.write(seq + "\n") # Write the row to the output file 
    
          print(f"Sequence file saved as: {output_file}")
    
      # Extract sequences
      extract_sequences(Homo_sapiens_input, Homo_sapiens_output, hg19_genome)
      extract_sequences(Gallus_gallus_input, Gallus_gallus_output, galGal4_genome)
      extract_sequences(Mus_musculus_input, Mus_musculus_output, mm10_genome)
    
    def extract_sequences_from_CircleBase():
      # Load reference genomes
      hg19_genome = Fasta('data/genomes/hg19.fa')
    
      # Load reference sequences
      Homo_sapiens_input = "data/raw/CircleBase/Homo_sapiens.tsv"
    
      # Save sequences
      os.makedirs("data/preprocessed/CircleBase", exist_ok=True)
    
      Homo_sapiens_output = "data/preprocessed/CircleBase/Homo_sapiens.txt"
    
      # Function to extract sequences and write to a new file
      def extract_sequences(input_file, output_file, genome):
          with open(input_file, newline='', encoding='utf-8') as infile, \
              open(output_file, "w", newline='', encoding='utf-8') as outfile:
    
              reader = csv.DictReader(infile, delimiter='\t')
    
              for row in tqdm(reader, desc=f"Extracting {input_file}"):
                  try:
                      chrom = row['chr_hg19']  # Extract chromosome name
                      start = int(float(row['start_hg19'])) - 1  # Convert start position to integer (0-indexed)
                      end = int(float(row['end_hg19']))  # Convert end position to integer
                      seq = genome[chrom][start:end].seq.upper()  # Extract sequence from genome file
                      if 'N' in seq or end - start > 10000:  # Skip sequences with 'N' (unknown bases) and sequences too long
                          continue
                  except Exception as e:
                      continue
    
                  outfile.write(seq + "\n") # Write the row to the output file 
    
          print(f"Sequence file saved as: {output_file}")
    
      # Extract sequences
      extract_sequences(Homo_sapiens_input, Homo_sapiens_output, hg19_genome)
    
    # Extract sequences from datasets
    extract_sequences_from_eccDNA_Atlas()
    extract_sequences_from_CircleBase()

extract_sequences()
