#!/usr/bin/env python

from analysis_utils.root import pytree
import ROOT as r

class reco_output:
    def __init__(self, output_filename, min_hits, **kwargs):
        self.outfile = r.TFile(output_filename, 'recreate')
        self.min_hits = min_hits

        for arg in ('eff','density','energy','theta','phi'):
            setattr(self, arg, kwargs.get(arg, 0))

        t = pytree.PyTree('reco', 'reco')
        self.t = t

        self.h = r.TH1I('nhits', 'nhits', 1000, 0, 1000)

        #self.write_result([])

    def write_result(self, hits):
        nhits = len(hits)
        self.h.Fill(nhits)

        if nhits < self.min_hits:
            return

        t = self.t

        t.reset()
        t.write_branch(self.eff, 'eff', float)
        t.write_branch(self.density, 'density', float)
        t.write_branch(self.energy, 'energy', float)
        t.write_branch(self.theta, 'theta', float)
        t.write_branch(self.phi, 'phi', float)
        t.write_branch(nhits, 'hit_n', float)
        t.write_branch(hits[:,0], 'hit_x', [int])
        t.write_branch(hits[:,1], 'hit_y', [int])
        t.write_branch(hits[:,2], 'hit_val', [int])
        t.fill()

    def close(self):
        self.outfile.Write()
        self.outfile.Close()
