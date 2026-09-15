from datasets import load_dataset

# Download the Databricks Dolly 15K dataset
dataset = load_dataset("databricks/databricks-dolly-15k")

# Display information about the dataset
print("Dataset information:")
print(dataset)

# Display the column names
print("\nColumn names:")
print(dataset["train"].column_names)

# Display the first example
print("\nFirst example:")
print(dataset["train"][0])