#!/usr/bin/env python

import sys
import numpy as np
from scipy.optimize import minimize, basinhopping
import generate

def expectations(ldf, grid, Ae, T):
    # calculate expected number of hits on each device for
    # the given shower pdf, detector postions, efficiencies
    # and coincidence interval.
    noise = 1e4/60*T*Ae * np.ones(len(grid))
    if ldf==None:
        return noise
    return Ae*np.array([ldf(x,y) for x,y in grid]) + noise

def llhood(ldf, hits, nohits, Ae, T):
    p1 = np.log(1.-np.exp(-1.*expectations(ldf, hits, Ae, T)))
    p0 = -1.*expectations(ldf, nohits, Ae, T)
    return np.sum(p0) + np.sum(p1)

def shower_fit(E0, density, grid_size, Ae, T):
    # generate random array of detectors
    grid = generate.make_detector_array(density, grid_size)
    # create the true LDF for the shower
    if E0:
        ldf0 = generate.make_shower_ldf(E0)
    else:
        ldf0 = None

    # draw the true number of hits per phone
    nhits = np.random.poisson(expectations(ldf0, grid, Ae, T))
    
    # separate phones that were hit or not hit
    hits = grid[nhits>0]
    nohits = grid[nhits==0]

    #print "saw %d hits" % len(hits)
    #print "true llhood:", llhood(ldf0, hits, nohits, Ae, T)

    def fn(xargs):
        logE, x0, y0 = xargs
        X0 = np.array([x0,y0])
        ldf = generate.make_shower_ldf(10**np.float(logE))
        return -2*llhood(ldf, hits+X0, nohits+X0, Ae, T)

    #print "optimize result:"
    res = minimize(fn, [16.0, 0, 0], tol=1e-3)
    if not res.success:
        print >> sys.stderr, res
    efit = 10**(res.x[0])
    x,y = res.x[1:]
    return efit, x, y

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="perform likelihood fitting on simulated shower")
    parser.add_argument("--density", type=float, default=1000, help="detector density [dev/km^2]")
    parser.add_argument("--size", type=float, default=500, help="grid size [m]")
    parser.add_argument("--Ae", type=float, default=5e-5, help="effective area A*epsilon [m^2]")
    parser.add_argument("-T", type=float, default=2, help="coincidence window [s]")
    parser.add_argument("-E", type=float, default=1e20, help="primary shower energy [eV] (none for no shower)")
    parser.add_argument("-N", type=int, default=1, help="number of trials")
    parser.add_argument("--out", help="output filename")
    parser.add_argument("--plot", action="store_true", help="show plot of results as the come in")
    args = parser.parse_args()

    import pylab as pl
    pl.ion()

    energies = []
    distances = []
    results = []
    for i in xrange(args.N):
        print "Generating %d/%d" % (i+1, args.N)
        efit, x, y = shower_fit(args.E, args.density, args.size, args.Ae, args.T)
        results.append([efit, x, y])
        #print i, efit, x, y

        if not args.plot: continue
        energies.append(efit)
        dist = np.sqrt(x**2+y**2)
        distances.append(dist)
        pl.figure(0)
        pl.clf()
        pl.hist(np.array(energies)/args.E, bins=25)
        pl.draw()
        pl.figure(1)
        pl.clf()
        pl.hist(distances, bins=25)
        pl.draw()

    if args.plot:
        raw_input("press enter")

    if args.out:
        import pandas as pd
        df = pd.DataFrame(results, columns=['E', 'x', 'y'])
        df.to_hdf(args.out, 'fits', mode='w')
        #np.save(args.out, results)
