"""
Selective launch script.

Usage: python script/selective_launch.py <port> <ranks> <ranks> <ranks> ...
"""
import os
import sys


def parse_ranks(ranks_strs):
    """
    parse_ranks
    """
    # NOTE: You can return ranks directly here to change script/train_gpu.sh
    # and script/kill_process.sh together
    #return [22, 23, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47]
    #return [22, 23, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47]
    #return [17, 19, 20, 22, 23, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43]
    #return [24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 51, 52, 53, 54, 55, 56]
    #return [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 56, 57, 58, 59, 61]
    #return [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59]

    # Example 1: Use contiguous nodes [8, 16)
    #return list(range(17, 32)) + [37]
    return range(0, 16)

    # Example 2: Use non-contiguous nodes [4, 8) + {10} + [30, 32), i.e., [4, 5, 6, 7, 10, 30, 31]
    #return list(range(4, 8)) + [10] + list(range(30, 32))

    # Example 3:
    # Just Python code, return any nodes you want!
    if not ranks_strs:
        return None

    ranks = []
    for r in ranks_strs:
        r = eval(r)
        if isinstance(r, int):
            ranks.append(r)
        else:
            ranks.extend(r)
    return ranks


def main(port, ranks):
    """
    main
    """
    ips = [ip.strip() for ip in os.getenv("TRAINER_INSTANCES").split(",") if ip.strip()]
    if ranks is None:
        ranks = list(range(len(ips)))
    ranks = sorted(list(set(ranks)))
    my_rank = int(os.getenv("POD_INDEX", "0"))
    if my_rank not in ranks:
        return

    rank = ranks.index(my_rank)
    nranks = len(ranks)

    master = ips[ranks[0]]
    print(f"--master {master}:{port} --rank {rank} --nnodes {nranks}")


if __name__ == "__main__":
    main(int(sys.argv[1]), parse_ranks(sys.argv[2:]))