import csv
import re

def txt_to_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # Write header (28 columns)
        header = ["ID", "Col2", "Col3", "Channel"] + [f"Val{i}" for i in range(1, 25)]
        writer.writerow(header)
        
        for line in infile:
            # First, insert a space before any '-' that follows a digit
            line = re.sub(r'(?<=\d)-', ' -', line)
            # Remove extra spaces and split into tokens
            tokens = re.split(r'\s+', line.strip())
            
            # First three tokens: ID, Col2, Col3
            row = tokens[:3]
            
            # The fourth token contains channel merged with number sometimes (e.g., "R-0.82")
            channel_match = re.match(r'([RGB])(.+)?', tokens[3])
            if channel_match:
                channel = channel_match.group(1)
                row.append(channel)
                if channel_match.group(2):
                    row.append(channel_match.group(2))
            else:
                row.append(tokens[3])
            
            # Add the rest of the numbers
            row.extend(tokens[4:])
            
            # Ensure exactly 28 columns (pad with blanks if short)
            while len(row) < 28:
                row.append("")
            
            writer.writerow(row[:28])

# Example usage:
txt_to_csv("input.txt", "output.csv")