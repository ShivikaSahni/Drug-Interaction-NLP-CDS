import xml.etree.ElementTree as ET
import csv

# Function to parse the XML and extract interactions
def parse_drugbank(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Namespace handling
    namespace = {"db": root.tag.split("}")[0].strip("{")}

    # Output list
    interactions_data = []

    # Drug counters
    total_drugs = 0
    drugs_with_interactions = 0
    drugs_without_interactions = 0

    for drug in root.findall("db:drug", namespace):
        total_drugs += 1

        # Extract basic drug information
        drug_id = drug.find("db:drugbank-id", namespace).text
        drug_name = drug.find("db:name", namespace).text

        # Extract interactions
        interactions = drug.find("db:drug-interactions", namespace)
        if interactions is not None and len(interactions) > 0:
            drugs_with_interactions += 1
            for interaction in interactions.findall("db:drug-interaction", namespace):
                interacting_drug_id = interaction.find("db:drugbank-id", namespace).text
                interaction_drug = interaction.find('db:name', namespace).text
                interaction_description = interaction.find("db:description", namespace).text
                interactions_data.append([drug_id, drug_name, interacting_drug_id, interaction_drug, interaction_description])
        else:
            drugs_without_interactions += 1

    # Print summary
    print(f"Total drugs: {total_drugs}")
    print(f"Drugs with interactions: {drugs_with_interactions}")
    print(f"Drugs without interactions: {drugs_without_interactions}")

    # Return extracted data
    return interactions_data

# Save to CSV
def save_to_csv(data, output_file):
    headers = ["Drug ID", "Drug Name", "Interacting Drug ID", "Interacting Drug Name", "Interaction Description"]
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)
    print(f"Data saved to {output_file}")

# Main
file_path = "./full database.xml"  # Replace with your file path
output_file = "./interaction_data.csv"

interactions = parse_drugbank(file_path)
save_to_csv(interactions, output_file)