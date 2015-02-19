#!/usr/bin/env python

import scipy as sci
import numpy as np
import pylab as pl
import time

MAX_HIT = 0
##############################################################
# simulation parameters (most are adjustable via command-line)
##############################################################

# size of the field over which to simulate. 1km seems to work okay,
# probaly could be smaller
GRID_WIDTH = 1000 # m

##################
# helper functions
##################

'''
return an array of random (x,y) pairs to represent
the detector positions.

detector density in N/km^2
width in m
'''
def make_detector_array(density, width=GRID_WIDTH):
    Nphones = density * (width/1e3)**2
    return np.random.uniform(-width/2.0, width/2.0, (Nphones,2))

'''
eta function, see EAS eqns. (8.14), (8.15)
'''
def eta_fn(th, N):
    return 3.88 - 0.64*(1.0/np.cos(th) - 1) + 0.07*np.log(N/1e8)

'''
construct and return a function that give a shower LDS
as a function of x and y.
See EAS eqns. (8.13) and others mentioned inline

arguments:
    energy - primary energy in eV
    theta - zenith angle in rad
    phi - azimuthal angle in rad
    s - shower age parameter
'''
def make_shower_ldf(energy, theta=0, phi=0, s=1):
    from scipy.special import gamma as gamma_fn
    # critical energy in air, see EAS (4.17) and discussion
    # on p.154.
    E_critical = 84e6 # eV

    # shower size is proportial to energy,
    # see EAS (4.88)
    N = energy/E_critical

    # HACK:
    # for muons, the normalization is about 3 orders lower
    # than for photons. would be better to get a muon-specific
    # LDF.
    N *= 1e-3
    
    eta = eta_fn(theta, N)
    alpha = 2-s

    # molliere radius, EAS p. 388
    rM = 100 # m

    # calculate the yucky gamma function term up front:
    gamma_factor = gamma_fn(eta-alpha) / (gamma_fn(2-alpha)*gamma_fn(eta-2))

    # unit vector pointing in the direction of the shower core:
    n_hat = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    def ldf(x, y):
        p = np.array([x,y,0])
        # vector distance from point p to line
        d = (n_hat.dot(p))*n_hat - p
        r = np.sqrt(sum(d**2))
        return N / (2*np.pi*rM**2) * (r/rM)**(-alpha) * (1 + r/rM)**(-(eta-alpha)) * gamma_factor

    # have numpy vectorize the function so we can
    # give it whole arrays of (x,y) values.
    return np.vectorize(ldf)

'''
draw from a poisson distribution to calculate the number of
particles that hit each detector, given their positions and the LDF.

returns a list of (x,y) pairs for phones that had at least one hit.
'''
def get_hits(pdf, grid, eff):
    global MAX_HIT
    device_hits = []
    for x,y in grid:
        # get flux at sample point (particles / m^2)
        flux = pdf(x, y)
        # get expected hits (lambda, in the paper)
        exp_hits = flux * eff
        # now sample the actual hits from poisson distribution
        actual_hits = sci.random.poisson(exp_hits)
        if actual_hits >= 1:
            device_hits.append((x,y))
            if actual_hits>MAX_HIT:
                MAX_HIT = actual_hits
    return np.array(device_hits)

'''
make a plot showing the location of phones; phones
with hits are highlighted in red.

If the overlay (X,Y,Z) grid is provided, also draw
a contour of the LDF.
'''
def pretty_plot(grid, hits, overlay=None):
    pl.clf()

    pl.scatter(grid[:,0],grid[:,1], color='skyblue')

    if len(hits):
        pl.scatter(device_hits[:,0], device_hits[:,1], color='red')

    # overlay the pdf contour
    if overlay:
        X, Y, Z = overlay
        pl.contour(Z, extent=[np.min(X), np.max(X), np.min(Y), np.max(Y)], linewidths=2.0)

    # label the axes
    pl.xlabel('meters')
    pl.ylabel('meters')

'''
Generate the an X,Y mesh grid, and sample the pdf
over it; used to make contour plots
'''
def make_overlay(pdf):
    X = np.linspace(-GRID_WIDTH/2.0, GRID_WIDTH/2.0, 100)
    Y = np.linspace(-GRID_WIDTH/2.0, GRID_WIDTH/2.0, 100)
    X,Y = np.meshgrid(X,Y)
    Z = np.log(pdf(X, Y))
    return (X,Y,Z)

###########################
# main command line program
###########################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate pseudo-experiments of phone hits from LDF")
    parser.add_argument('--out', help='output filename')
    parser.add_argument('-s', '--seed', type=int, help='Use a specific seed')
    parser.add_argument('-i', '--interactive', action='store_true', help='Plot in interactive mode.')
    parser.add_argument('--nhits', type=int, default=10, help='minimum number of coincident detector hits required for events')
    parser.add_argument('--nevents', type=int, default=10000, help='number of events to generate')
    parser.add_argument('--eff', type=float, default=1e-4, help='the effective area (efficiency*A) of the individual phones [in m^2]')
    parser.add_argument('-N', '--ndetectors', type=int, default=1000, help='the number of detectors per km^2')
    parser.add_argument('--age', type=float, default=1.8, help='shower age parameter')
    parser.add_argument('--theta', type=float, default=0, help='zenith angle of incident particle')
    parser.add_argument('--phi', type=float, default=0, help='azimutal angle of incident particle')
    parser.add_argument('--energy', default=1e19, type=float, help='the energy (in eV) of the primary particle')

    args = parser.parse_args()

    # set the seed if the user wants to specify one
    if args.seed == None:
        seed = int(time.time())
    else:
        seed = args.seed

    if args.interactive:
        pl.ion()

    # set up the initial detector grid
    np.random.seed(seed)

    # construct the LDF for the given shower parameters
    ldf = make_shower_ldf(energy=args.energy, theta=args.theta, phi=args.phi, s=args.age)

    if args.interactive:
        # sample the LDF on a grid so we can
        # make a pretty contour overlay
        overlay = make_overlay(ldf)

    # keep track of the number of samples generated,
    # and the number kept for "reco" (i.e. those which
    # have at least the minimum number of detector hits)
    total_samples = 0
    reco_samples = 0

    if args.out:
        from output import reco_output
        output = reco_output(args.out, args.nhits)
        output.eff = args.eff
        output.density = args.ndetectors
        output.energy = args.energy
        output.theta = args.theta
        output.phi = args.phi

    update_interval = args.nevents/10
    start_time = time.time()
    while True:
        if not args.interactive and args.nevents>0 and total_samples >= args.nevents:
            # we're done here!
            break

        if total_samples%update_interval==0:
            print "Generating %d / %d" % (total_samples, args.nevents)

        # regenerate the random phone grid
        grid = make_detector_array(args.ndetectors)

        device_hits = get_hits(ldf, grid, args.eff)
        total_samples += 1
        
        if args.out:
            output.write_result(device_hits)

        if len(device_hits) < args.nhits:
            # cut this event, and start over
            continue
        reco_samples += 1

        print "%d / %d shower hits (%.2f%% eff)" % (reco_samples, total_samples, 100.*reco_samples/total_samples)

        if args.interactive:
            pretty_plot(grid, device_hits, overlay)

            action = None
            while not action in ('','c','q',):
                print "[C]ontinue, [q]uit? ",
                action = raw_input().lower()
            if action == 'q':
                # we're done here.
                break

    if args.out:
        output.close()

    print "Done. Generated %d events in %ds" % (reco_samples, time.time()-start_time)
    print "Generator efficiency: %.2f%%" % (100.*reco_samples/total_samples)
