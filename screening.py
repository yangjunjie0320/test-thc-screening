import pyscf, numpy, scipy
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

def build_sparse_rho(phi=None, tol=1e-8):
    ng, nao = phi.shape

    from scipy.sparse import dok_array
    rho_s = dok_array((ng, nao * (nao + 1) // 2))
    mask = numpy.where(numpy.abs(phi) > tol) # numpy.sqrt(tol))

    # Do we have a smarter way of doing this?
    for ig, mu in zip(*mask):
        rho_g_mu = phi[ig, mu] * phi[ig, :(mu+1)]

        for nu in range(mu+1):
            munu = mu * (mu + 1) // 2 + nu

            if abs(rho_g_mu[nu]) > tol:
                rho_s[ig, munu] = rho_g_mu[nu]

    return rho_s

if __name__ == "__main__":
    for n in [2, 4, 8, 16, 32, 64]:
        m = WaterCluster(n=n, basis="631g*", verbose=0)
        m.max_memory = 400

        from pyscf.dft import Grids
        from pyscf.dft.numint import NumInt
        ni   = NumInt()
        grid = Grids(m)
        grid.atom_grid = {"O": (19, 50), "H": (11, 50)}
        grid.prune = None
        grid.build()

        from pyscf.lib.logger import perf_counter, process_clock
        log = pyscf.lib.logger.Logger(verbose=5)
        
        xao = numpy.einsum("Ru,R->Ru", ni.eval_ao(m, grid.coords), grid.weights ** 0.5)
        ngrid, nao = xao.shape

        t0 = process_clock(), perf_counter()
        rho = pyscf.lib.pack_tril(numpy.einsum("Ru,Rv->Ruv", xao, xao))
        t0 = log.timer("calculate rho", *t0)

        rho_s = build_sparse_rho(xao, tol=1e-6) # What is the smarter way of doing this?
        t0 = log.timer("calculate rho_s", *t0)

        ovlp_s = rho_s.dot(rho_s.T)
        t0 = log.timer("calculate ovlp_s", *t0)

        # # rho = numpy.einsum("Ru,Rv->Ruv", xao, xao)
        # # ngrid = rho.shape[0]
        # # nao = rho.shape[1]

        # # idx = numpy.arange(nao)
        # # rho_tril = pyscf.lib.pack_tril(rho) # + rho.transpose(0,2,1))
        # # rho_tril[:, idx*(idx+1)//2+idx] *= 0.5
        
        # df = pyscf.df.DF(m)
        # df.max_memory = 400
        # df.auxbasis = "weigend"
        # df.build()
        # naux = df.get_naoaux()
        # t0 = log.timer("density fitting", *t0)
        

        # max_memory = 400
        # jj = numpy.zeros((naux, ngrid))
        # blksize = max(4, int(min(df.blockdim, max_memory * 3e5 / 8 / nao**2)))

        # p1 = 0
        # for istep, chol in enumerate(df.loop(blksize=blksize)):
        #     p0, p1 = p1, p1 + chol.shape[0]
        #     jj[p0:p1] = rho_s.dot(chol.T).T

        # t0 = log.timer("THC", *t0)
