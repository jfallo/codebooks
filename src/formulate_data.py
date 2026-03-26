import pandas as pd
import os

df = pd.read_csv("./intermediate/codebooks_metadata.csv", sep= '|')
df = df.drop_duplicates()
df.to_csv("./intermediate/codebooks_metadata.csv", index= False, sep= '|')

df = df.drop(columns= ["index"])

df["ICPSR"] = df["study"].str.extract(r"\(ICPSR\s+(\d+)\)").astype(int)
df["DS"] = df["dataset"].str.extract(r"DS\s+(\d+)").astype(int)
df["id"] = df["ICPSR"].astype(str).str.zfill(5) + '-' + df["DS"].astype(str).str.zfill(4)
df = df.sort_values(by= ["ICPSR", "DS"], ascending= True)

files = os.listdir("./intermediate")
def find_codebook(id):
    prefix = id.split('-')[0]
    exact_match = None
    prefix_match = None

    for file in files:
        if id in file:
            exact_match = file
        elif file.startswith(prefix):
            prefix_match = file

    if exact_match:
        return os.path.join("./intermediate", exact_match)
    if prefix_match:
        return os.path.join("./intermediate", prefix_match)
    return None

df["codebook"] = df["id"].apply(find_codebook)

df = df.drop_duplicates()
df = df[["id", "ICPSR", "DS", "codebook", "study", "dataset", "url"]]
df.to_csv("./output/codebooks_metadata.csv", index= False, sep= '|')

