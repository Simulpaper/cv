import cv2
import itertools

from component_classification import get_component_classifications
from generate_subimages import get_edges_subimages
from parse_dataset import get_dataset

def generate_circuits(new_edges):
    # generate every possible combination of circuits and calculate their scores
    edge_types = [edge[2] for edge in new_edges]
    circuit_combos = list(itertools.product(*edge_types))

    print("\n")
    # print(circuit_combos)
    print(len(circuit_combos))

    possible_circuits = []
    for circuit_combo in circuit_combos:
        circuit = []
        score = 0
        for i, (node1, node2, _) in enumerate(new_edges):
            circuit.append([node1, node2, circuit_combo[i][0]])
            score += circuit_combo[i][2]
        # calculate likelihood score here and add it
        possible_circuits.append([circuit, score])
    
    possible_circuits.sort(key = lambda x: x[1])

    return possible_circuits
    

if __name__ == "__main__":
    # list of [circle1, circle2, subimages]
    edge_subimages = get_edges_subimages("component_images/demo.jpg")

    bil_params = (5, 25, 25)
    # Setting parameter values for Canny
    t_lower = 100 # Lower Threshold
    t_upper = 300 # Upper threshold
    orb = cv2.ORB_create()
    dataset = get_dataset(orb, bil_params, t_lower, t_upper)
    new_edges = [] #list of [n1_index, n2_index, classifications]

    for i in range(len(edge_subimages)):
        subimage = edge_subimages[i][2]
        # classifications = (component type, num matches, avg dist)
        classifications = get_component_classifications(orb, bil_params, t_lower, t_upper, subimage, dataset)
        new_edges.append([edge_subimages[i][0], edge_subimages[i][1], classifications])
    
    possible_circuits = generate_circuits(new_edges)
    
    for i in range(5):
        print(possible_circuits[i])
        print("\n")

    # take the top five and create netlists for each one


