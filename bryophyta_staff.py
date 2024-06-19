from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Any

import numpy as np
import scipy.stats as stats

p_moss_giant_not_task = 1 / 150
p_moss_giant_task = 1 / 50
p_staff = 1 / 118
p_bryo_key = 1 / 16
p_skot_pet = 1 / 65
p_ancient_shard = 1 / 293
p_totems = 1 / 440
SKOTIZO_PET: bool = False


def kill_bryophyta(current_kc) -> bool:
    if stats.geom.rvs(p_staff) == 1:
        return True
    if stats.geom.rvs(p_bryo_key) == 1:
        kill_bryophyta(current_kc)


def kill_skotizo():
    if stats.geom.rvs(p_skot_pet) == 1:
        SKOTIZO_PET = True


def simulate_one_staff_drop(droprate) -> (int, int, int):
    kills = 0
    skotizo_count = 0
    ancient_shards = 0
    totems = 0
    staff_dropped = False

    while not staff_dropped:
        kills_until_key = stats.geom.rvs(droprate)

        ac = stats.geom.rvs(p_ancient_shard, size=kills_until_key)
        tm = stats.geom.rvs(p_totems, size=kills_until_key)
        ancient_shards += np.sum(ac == 1)
        totems += np.sum(tm == 1)
        if totems >= 3:
            skotizo_count += 1
            kill_skotizo()
            totems -= 3
        kills += kills_until_key
        staff_dropped = kill_bryophyta(kills)

    return kills, skotizo_count, ancient_shards


def simulate(drop_rate, simulations) -> list:
    results = list()
    with ThreadPoolExecutor(10) as executor:
        futures = {executor.submit(simulate_one_staff_drop, drop_rate): i
                   for i in range(simulations)}

        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as ex:
                print(f'Exception: {ex}')
                exit(1)

    return results


if __name__ == '__main__':
    print('Bryophyta\'s Essence Simulator')
    print('1) Not on Slayer Task\n2) On Slayer Task')
    inp = int(input('Enter: '))
    drop_rate = p_moss_giant_not_task if inp == 1 else p_moss_giant_task

    result = simulate(drop_rate, 10000)
    giants = [x[0] for x in result]
    skotizo = [x[1] for x in result]
    ancient_shards = [x[2] for x in result]

    mean_giant_kc = np.mean(giants)
    mean_skotizo = np.mean(skotizo)
    mean_ancient_shards = np.mean(ancient_shards)

    print('Mean Moss Giant KC: ', mean_giant_kc)
    print('Mean Ancient Shards: ', mean_ancient_shards)
    print('Mean Skotizo: ', mean_skotizo)
    if SKOTIZO_PET:
        print('You have a funny feeling like you\'re being followed.')