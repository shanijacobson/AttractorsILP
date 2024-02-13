import boolean_network as bn
import time
import ilp


TIMEOUT_IN_SEC = 10_800


def main(network_name):
    start_time = time.time()
    print(f"Start to search attractors for network: {network_name}")
    network = bn.BooleanNetwork(network_name)
    ilp.find_stable_states_and_external(network)
    total_time = time.time() - start_time
    print(f"Finish to run network: {network_name}. Total time: {total_time}.")