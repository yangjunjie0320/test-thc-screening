import pyscf, numpy, scipy
from scipy.sparse import dok_array

from pyscf import gto, scf, ao2mo
from pyscf.lib import logger

def WaterCluster(n=64, basis="ccpvdz", verbose=4):
    atom = ""
    with open("./64h2o.xyz", "r") as f:
        lines = f.readlines()
        atom = "".join(lines[2:(2+3*n)])

    mol = gto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.max_memory=4000
    mol.verbose = verbose
    mol.build()

    return mol

def chole_decomp(phi, tol=1e-8):
    tol = min(tol, tol ** 2, 1e-6)
    if tol < 1e-12:
        tol = 1e-12

    ngrid, nao = phi.shape
    rho = dok_array((ngrid, nao * (nao + 1) // 2))
    mask = numpy.where(numpy.abs(phi) > numpy.sqrt(tol))

    for ig, mu in zip(*mask):
        rho_g_mu = phi[ig, mu] * phi[ig, :(mu+1)]
        rho_g_mu[-1] /= numpy.sqrt(2)

        munu = mu * (mu + 1) // 2 + numpy.arange(mu+1)
        ix = numpy.abs(rho_g_mu) > tol
        rho[ig, munu[ix]] = rho_g_mu[ix]

    ss  = rho.dot(rho.T)
    ss += ss.T

    from scipy.linalg.lapack import dpstrf
    chol, perm, rank, info = dpstrf(ss.todense(), tol=tol)
    assert info == 1  # Make sure pivoting Cholesky runs in success

    nispt = rank
    perm = (numpy.array(perm) - 1)[:nispt]

    tril = numpy.tril_indices(nispt, k=-1)
    chol = chol[:nispt, :nispt]
    chol[tril] *= 0.0
    visp = phi[perm]
    return chol, visp

def build_rho(phi=None, tol=1e-8):
    tol = min(tol, tol ** 2, 1e-6)
    if tol < 1e-12:
        tol = 1e-12

    ng, nao = phi.shape
    rho = dok_array((ng, nao * (nao + 1) // 2))

    # This part remains similar; identifying non-zero elements
    mask = numpy.where(numpy.abs(phi) > numpy.sqrt(tol))

    for ig, mu in zip(*mask):
        rho_g_mu = phi[ig, mu] * phi[ig, :(mu+1)]
        rho_g_mu[-1] *= 0.5

        # Vectorize or optimize this loop
        munu = mu * (mu + 1) // 2 + numpy.arange(mu+1)
        ix   = numpy.abs(rho_g_mu) > tol
        rho[ig, munu[ix]] = rho_g_mu[ix]

    return rho

if __name__ == "__main__":
    tol = 1e-4
    for n in [2]:
        m = WaterCluster(n=n, basis="631g*", verbose=0)
        m.max_memory = 400

        from pyscf.dft import Grids
        from pyscf.dft.numint import NumInt
        ni   = NumInt()
        grid = Grids(m)
        grid.atom_grid = {"O": (19, 50), "H": (11, 50)}
        grid.prune = None
        # grid.level = 1
        grid.build()

        from pyscf.lib.logger import perf_counter, process_clock
        log = pyscf.lib.logger.Logger(verbose=5)
        
        phi  = ni.eval_ao(m, grid.coords)
        phi *= (numpy.abs(grid.weights) ** 0.5)[:, None]
        chol, visp = chole_decomp(phi, tol=tol) # How to setup the tolerance?
        nisp, nao = visp.shape
        rho = build_rho(visp, tol=tol)
        
        df = pyscf.df.DF(m)
        df.max_memory = 400
        df.auxbasis = "weigend"
        df.build()
        naux = df.get_naoaux()

        coul = numpy.zeros((naux, nisp))

        p1 = 0
        blksize = 10
        
        for istep, chol_l in enumerate(df.loop(blksize=blksize)):
            p0, p1 = p1, p1 + chol_l.shape[0]
            coul[p0:p1] = rho.dot(chol_l.T).T * 2.0

        ww = scipy.linalg.solve_triangular(chol.T, coul.T, lower=True).T
        vv = scipy.linalg.solve_triangular(chol, ww.T, lower=False).T

