'''
author:     Alexandros E. Ntigkaris
python:     3.9.2
numpy:      1.19.5
matplotlib: 3.3.4
'''

if __name__ == '__main__':

    from utils import BatColony

    params = {
                'entities':15,
                'timesteps':10,
                'alpha':0.9,
                'gamma':0.9,
                'benchmark_fn':'dejong',
                'random_state':2036,
                'sleep_rate':1e-10,
                'verbose':True,
            }

    bats = BatColony(**params)
    bats.fill()
    bats.run()

