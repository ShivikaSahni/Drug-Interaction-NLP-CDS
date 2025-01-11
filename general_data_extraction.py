import pandas as pd
import xml.etree.ElementTree as ET

# Path to the DrugBank XML file
drugbank_file = './full database.xml'

# Parse the XML file
tree = ET.parse(drugbank_file)
root = tree.getroot()

# Namespace handling
ns = {'db': 'http://www.drugbank.ca'}

# Initialize lists to store extracted data
drug_data = []

# Extract relevant drug information
for drug in root.findall('db:drug', ns):
    drug_id = drug.find('db:drugbank-id', ns).text if drug.find('db:drugbank-id', ns) is not None else None
    name = drug.find('db:name', ns).text if drug.find('db:name', ns) is not None else None
    description = drug.find('db:description', ns).text if drug.find('db:description', ns) is not None else None
    indication = drug.find('db:indication', ns).text if drug.find('db:indication', ns) is not None else None
    pharmacodynamics = drug.find('db:pharmacodynamics', ns).text if drug.find('db:pharmacodynamics', ns) is not None else None

    # Extract drug interactions
    interactions = drug.findall('db:drug-interactions/db:drug-interaction', ns)
    interaction_list = []
    for interaction in interactions:
        interaction_drug = interaction.find('db:name', ns).text if interaction.find('db:name', ns) is not None else None
        interaction_list.append(interaction_drug)
    interactions_str = ', '.join(interaction_list)

    # Append data to list
    drug_data.append({
        'Drug ID': drug_id,
        'Name': name,
        'Description': description,
        'Indication': indication,
        'Pharmacodynamics': pharmacodynamics,
        'Interactions': interactions_str
    })

# Convert to a DataFrame
df = pd.DataFrame(drug_data)

# Save to CSV
output_file = './general_data.csv'
df.to_csv(output_file, index=False)

print(f"Data has been successfully extracted and saved to {output_file}")
