import pandas as pd

# Read Excel and save as CSV
df = pd.read_excel("data/Online Retail.xlsx")
df.to_csv("data/customer_data.csv", index=False)
print("Successfully saved as customer_data.csv!")