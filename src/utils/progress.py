from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib


def parallel_progress(func, iterable, n_jobs=1, total=None, desc="Progresso", unit="it"):
    """
    Executa func em paralelo sobre iterable, mostrando uma barra de progresso única em tempo real.
    """
    results = []
    with tqdm(total=total or len(iterable), desc=desc, unit=unit) as pbar:
        with tqdm_joblib(pbar):
            results = Parallel(n_jobs=n_jobs)(delayed(func)(item) for item in iterable)
    return results 