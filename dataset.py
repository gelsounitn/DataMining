import argparse

def main(args):
    dataset = args.d

    if type(dataset) != type(""):
        raise TypeError("The argument --d is not a string")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d', type = str, required = True)

    main(args = parser.parse_args())