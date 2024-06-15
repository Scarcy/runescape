import numpy as np
import scipy.stats as stats
from matplotlib.lines import Line2D
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt


def simulate_one_run(target_pearls: int, probability: float):
    """
    Function that the ThreadPool Threads will execute.
    This function will be executed in parallel.

    """
    total_pearls: int = 0
    chests_opened: int = 0

    while total_pearls < target_pearls:
        # RVS returns a random amount of chests needed to get one pearl drop
        # This saves me from having a 2D Loop which is significantly slower
        chests_opened += stats.geom.rvs(p=probability)
        # Since Pearl Drops range from 10 to 20, this gives a random pearl amount in that range
        total_pearls += stats.randint.rvs(10, 21)

    return chests_opened


def run_threaded(target_pearls: int, probability: float, simulations: int = 10_000,
                 maxworkers: int = None) -> np.ndarray:
    results = []
    completed_simulations = 0
    # Sets up the ThreadPool and tells the threads which function to run
    # When a thread is finished executing, it comes back to start another Simulation Run
    # This is means we can limit thread creation, since the same 8-10 threads work until the job is done.
    with ThreadPoolExecutor(max_workers=maxworkers) as executor:
        futures = {executor.submit(simulate_one_run, target_pearls, probability): i
                   for i in range(simulations)}
        # We get here when threads finish executing.
        for i, future in enumerate(as_completed(futures)):
            try:
                completed_simulations += 1
                # Add the result from the thread to the "results" list
                results.append(future.result())

                # A "Progress Bar". Prints out every 10k finished simulations.
                # This is because the runs take a very long time at 1m+ simulations
                if i % 10_000 == 0:
                    print(f"{completed_simulations} Simulations out of {simulations}...")
            except Exception as ex:
                print(f"Exception: {ex}")

        arr = np.array(results)

        return arr


def graph2(arr: np.ndarray, pearls: int, sims: int):
    """Creates the graph"""

    # Configures the image size
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Histogram bars from the lowest data point to the highest datapoint
    # Creates bars every 3 along the x axes.
    # This was an attempt to group similar values together
    bins_arr = [i for i in range(arr.min() - 3, arr.max() + 3, 3)]

    # Plots the Histogram
    n, bins, patches = ax.hist(arr, bins=bins_arr, density=True, histtype='stepfilled', alpha=0.2)

    ax.set_xlim(0)  # This might be deleted or changed. It's config for the x-axis
    sdev = np.std(arr) # Compute the Standard Deviation
    smean = np.mean(arr) # Compute the Mean

    # Y-Value for the Standard Deviation "dots"
    # This calculates the center of the graph
    center_y = max(abs(n)) / 2
    # Creates the vertical Mean Line
    ax.axvline(x=smean, label="Mean", color="red", zorder=5)
    # Creates the Standard Deviation line-dots
    ax.plot(
        [smean - sdev, smean + sdev], [center_y, center_y],
        'k:', lw=1.5, zorder=10,
        label='+/- Standard Deviation'
    )

    # Normal Distribution Plot
    mu, std = stats.norm.fit(arr)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, sims)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', lw=2, zorder=11)

    # Create custom legend handles
    custom_lines = [
        Line2D([0], [0], color='red', lw=2, label='Mean'),
        Line2D([0], [0], color='black', linestyle='dotted', lw=2, label='+/- Standard Deviation'),
        Line2D([0], [0], color='blue', label='Normal Distribution'),
        Line2D([0], [0], color='none', label=f'Simulations: {sims:,}')
    ]

    # Text Box with stats
    text_str = '\n'.join((
        f'Mean: {mean:.2f}',
        f'Std: {sdev:.2f}'
    ))
    box = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(
        0.05, 0.95, text_str, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', bbox=box
    )

    ax.legend(handles=custom_lines, loc='best', frameon=False)

    plt.suptitle("Guardians of the Rift")
    plt.title(f"Simulations to find how many Reward Chests to open for {pearls} Abyssal Pearls", fontsize=9)
    plt.show()

    print(f"CenterY: {center_y}\nSdev: {sdev}\nMean: {smean}")


# If another file imports this one, this code won't be executed
# It makes it possible to reuse the functions above without executing the code below
if __name__ == "__main__":
    NUM_THREADS = 8

    target_pearls: int = int(input("How many pearls?: "))
    probability: float = 1 / 6.94
    simulations: int = int(input("How many simulations?: "))
    result = run_threaded(target_pearls, probability, simulations=simulations)

    var = np.var(result)
    std = np.std(result)
    mean = np.mean(result)
    print(f"Mean: {mean:.1f}")
    print(f"Variance: {var:.1f}")
    print(f"Standard Deviation: {std:.1f}")

    graph2(result, pearls=target_pearls, sims=simulations)
