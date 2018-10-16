import argparse
import sys
from run_utils_class import RunLoop

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to config file.')

def main():
    args = parser.parse_args(sys.argv[1:])
    try: RunLoop(args.config)
    except ValueError as ve: print(ve)
    
if __name__ == '__main__':
    main()