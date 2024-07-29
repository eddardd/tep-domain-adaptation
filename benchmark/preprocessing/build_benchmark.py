import os
import h5py
import pickle
import numpy as np
from tqdm.auto import tqdm


def tep_build_benchmark(in_path='./tep_data/raw',
                        out_path='./tep_data/benchmark',
                        variables=None):
    """Builds the Tennessee Eastman (TE)
    Benchmark for domain adaptation.

    Parameters
    ----------
    in_path : string
        String containing the path towards the raw TE data.
    out_path  : string
        String containing the path to the directory where
        the pre-processed data will be saved.
    variables : list of integers (optional)
        List of integers within the range 0, ..., 53, specifying
        which variables to use for the benchmark. If not provided,
        or if None is specified, uses 1, ..., 23 and 43, ..., 53.
    """
    if variables is None:
        variables = [
            i for i in range(1, 24)] + [
                j for j in range(43, 54)]

    for mode in range(1, 7):
        print("Processing data from mode {}".format(mode))
        with h5py.File(
                os.path.join(
                    in_path, 'TEP_Mode{}.h5'.format(mode))) as hdf_file:
            Xnormal, Xfaulty, ynormal, yfaulty = [], [], [], []
            fault_intensity = -1
            for idv in range(1, 29):
                try:
                    runs = hdf_file[
                        f'Mode{mode}'][
                            'SingleFault'][
                                'SimulationCompleted'][
                                    f'IDV{idv}'][
                                        f'Mode{mode}_IDVInfo_{idv}_100']
                    fault_intensity = 100
                except KeyError:
                    try:
                        runs = hdf_file[
                            f'Mode{mode}'][
                                'SingleFault'][
                                    'SimulationCompleted'][
                                        f'IDV{idv}'][
                                            f'Mode{mode}_IDVInfo_{idv}_75']
                        fault_intensity = 75
                    except KeyError:
                        try:
                            runs = hdf_file[
                                f'Mode{mode}'][
                                    'SingleFault'][
                                        'SimulationCompleted'][
                                            f'IDV{idv}'][
                                                f'Mode{mode}_IDVInfo_{idv}_50']
                            fault_intensity = 50
                        except KeyError:
                            runs = hdf_file[
                                f'Mode{mode}'][
                                    'SingleFault'][
                                        'SimulationCompleted'][
                                            f'IDV{idv}'][
                                                f'Mode{mode}_IDVInfo_{idv}_25']
                            fault_intensity = 25

                print(
                    f'Reading IDV {idv} from Mode {mode}, '
                    f'Fault Intensity: {fault_intensity}')

                # Runs over simulations to get signal data
                pbar = tqdm(range(1, 101))
                for r in pbar:
                    flag = False
                    try:
                        array = runs['Run{}'.format(r)]['processdata'][:]
                        flag = True
                    except KeyError:
                        pbar.set_description(
                            "Unable to process run {}".format(r))
                    if flag:
                        Xnormal.append(array[None, :600, variables])
                        Xfaulty.append(array[None, 600:1200, variables])
                        ynormal.append(0)
                        yfaulty.append(idv)
                        pbar.set_description("Shape: {}".format(array.shape))

                # Sub-sampling normal class
                ind = np.arange(len(Xnormal))
                Xnormal = [Xnormal[i] for i in ind[:100]]
                ynormal = [ynormal[i] for i in ind[:100]]

                # Creating dataset
                X = np.concatenate(Xnormal + Xfaulty, axis=0)
                y = np.array(ynormal + yfaulty)

                # Creates pickle file
                dataset = {
                    "Signals": X, "Labels": y
                }

                with open(os.path.join(
                    out_path, f"TEPDataset_Mode{mode}.pickle"),
                          'wb') as f:
                    pickle.dump(dataset, f)
