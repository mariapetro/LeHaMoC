import sys
import simulation_params as sp
import LeMoC as LC


def main_cli():
    if len(sys.argv) != 2:
        print('incorrect parameters passed')
        print('try something like this')
        print('LeMoC ./Parameters.txt')
        quit()

    # Read parameters file
    fileName = sys.argv[1]

    simpam = sp.load_param_file(file_name=fileName)
    LC.run(simpam)




