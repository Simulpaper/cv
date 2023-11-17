import sys

def parse_dataset(filename):
    dataset = []

    with open(filename, "r") as file:
        lines = file.readlines()
        print(lines)
        for line in lines:
            components = line.strip().split(': ')
            component_type = components[0]
            descriptors = components[1]

            print(sys.getsizeof(descriptors))

            dataset.append((component_type, descriptors))

    return dataset

if __name__ == "__main__":
    dataset = parse_dataset("dataset_info.txt")

