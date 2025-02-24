import scripts
import scripts.gen_optimized_datasets
import sys

if __name__ == '__main__':
    if len(sys.argv) == '1':
        sys.argv.append("gen_optimized_datasets")
    if sys.argv[1] == "gen_optimized_datasets":
        scripts.gen_optimized_datasets.main()