"""

"""


# Built-in

# Libs
import scipy.signal
import toolman as tm
import matplotlib.pyplot as plt

# Own modules


def parse_line(log_line):
    data_log, loss_log = log_line.split(')')

    epoch = int(tm.misc_utils.get_digits(data_log.split(',')[0].split(':')[1]))
    iter_num = int(tm.misc_utils.get_digits(data_log.split(',')[1].split(':')[1]))

    loss_items = loss_log.strip().split(' ')
    loss_items = {a.split(':')[0]: float(b) for a, b in zip(loss_items[::2], loss_items[1::2])}

    return epoch, iter_num, loss_items


def main():
    # settings
    log_dir = r'./checkpoints/rs_cyclegan/loss_log.txt'
    keys = ['D_A', 'G_A', 'cycle_A', 'sem_A', 'D_B', 'G_B', 'cycle_B', 'sem_B']
    loss_lists = [[] for _ in range(len(keys))]

    log_file = tm.misc_utils.load_file(log_dir)[2:]
    for line in log_file:
        epoch, iter_num, loss_items = parse_line(line)
        for cnt, k in enumerate(keys):
            loss_lists[cnt].append(loss_items[k])

    # plot the loss curve
    plt.figure(figsize=(18, 6))

    for cnt, k in enumerate(keys):
        plt.subplot(2, 4, cnt+1)
        plt.plot(scipy.signal.savgol_filter(loss_lists[cnt], 9, 2))
        plt.title(k)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
