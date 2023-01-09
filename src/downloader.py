import sys
from pathlib import Path
import pandas as pd
from subprocess import run
from argparse import ArgumentParser
# # Python is not great for parallism, I like to use bash

def main(args):
    # Transform parquet file to data for bash script
    df = pd.read_parquet(args.datafile)
    with (__dir__ / "images.txt").open("w") as f:
        for idx, row in df.iterrows():
            f.write(f"images/{row['asin']}.jpg {row['primary_image']}\n")
    # Create bash script
    with (__dir__ / "download.sh").open("w") as f:
        f.write("#!/bin/bash\n")
        f.write("mkdir -p images\n")
        f.write(f"cat images.txt | xargs -n 2 -P {args.processes} curl -s -o \n")
        f.write("rm images.txt download.sh\n")
    # Run bash script
    if args.run:
        return run(["/bin/bash", str(__dir__ / "download.sh")], shell=True).returncode
    return 0

if __name__ == "__main__":
    __dir__ = Path(__file__).absolute().parent.parent
    argparse = ArgumentParser()
    argparse.add_argument("--run", type=bool, default=False)
    argparse.add_argument("--processes", type=int, default=8)
    argparse.add_argument("--datafile", type=str, default=str(__dir__/"data"/"product_images.parquet"))
    sys.exit(main(argparse.parse_args()))
