import os
import shutil

# Remove Chroma database directory if it exists
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print("Chroma database directory has been removed.")
else:
    print("Chroma database directory does not exist.")