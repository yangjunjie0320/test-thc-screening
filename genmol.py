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

def get_shell_pair_for_grid(mol, coords, rho=None):
    nao = mol.nao_nr()
    
    atom_coords = mol.atom_coords()
    ao_slice = mol.aoslice_by_atom()

    shell_pair = []
    for ig, coordg in enumerate(coords):
        shell_pair_on_ig = []
        for iatm in range(mol.natm):
            for jatm in range(mol.natm):
                coordi = atom_coords[iatm]
                coordj = atom_coords[jatm]
                
                rij = numpy.linalg.norm(coordi - coordj)
                rig = numpy.linalg.norm(coordi - coordg)
                rjg = numpy.linalg.norm(coordj - coordg)

                if numpy.max([rij, rig, rjg]) < 8.0:
                    pi_0, pi_1 = ao_slice[iatm, 2:4]
                    pj_0, pj_1 = ao_slice[jatm, 2:4]

                    
                    r = rho[ig, pi_0:pi_1, pj_0:pj_1]
                else:
                    pi_0, pi_1 = ao_slice[iatm, 2:4]
                    pj_0, pj_1 = ao_slice[jatm, 2:4]

                    # print("\n not selected", iatm, jatm, ig, pi_0, pi_1, pj_0, pj_1)
                    r = rho[ig, pi_0:pi_1, pj_0:pj_1]
                    assert numpy.linalg.norm(r) < 1e-4

def build_sparse_rho(coords, phi=None):
    ng, nao = phi.shape

    rho_s = []
    for ig, rg in enumerate(coords):
        rhog_s = {}

        for mu in range(nao):
            for nu in range(mu):
                if abs(phi[ig, mu] * phi[ig, nu]) > 1e-8:
                    # turn mu nu into the upper t
                    rhog_s[mu * (mu + 1) // 2 + nu] = phi[ig, mu] * phi[ig, nu]

        rho_s.append(rhog_s)
    return rhog_s


if __name__ == "__main__":
    for n in [2, 4, 8, 16, 32, 48, 64]:
        m = WaterCluster(n=n, basis="sto3g", verbose=0)
        m.max_memory = 400

        from pyscf.dft import Grids
        from pyscf.dft.numint import NumInt
        ni   = NumInt()
        grid = Grids(m)
        grid.atom_grid = {"O": (19, 50), "H": (11, 50)}
        grid.prune = None
        grid.build()

        xao = numpy.einsum("Ru,R->Ru", ni.eval_ao(m, grid.coords), grid.weights ** 0.5)
        rho = numpy.einsum("Ru,Rv->Ruv", xao, xao)
        ngrid = rho.shape[0]
        nao = rho.shape[1]

        idx = numpy.arange(nao)
        rho_tril = pyscf.lib.pack_tril(rho) # + rho.transpose(0,2,1))
        rho_tril[:, idx*(idx+1)//2+idx] *= 0.5

        from pyscf.lib.logger import perf_counter, process_clock

        log = pyscf.lib.logger.Logger(verbose=5)
        t0 = process_clock(), perf_counter()
        
        df = pyscf.df.DF(m)
        df.max_memory = 400
        df.auxbasis = "weigend"
        df.build()

        print("")
        t0 = log.timer("\ndensity fitting", *t0)

        naux = df.get_naoaux()

        max_memory = 400
        jj = numpy.zeros((naux, ngrid))
        blksize = max(4, int(min(df.blockdim, max_memory * 3e5 / 8 / nao**2)))
        # print("blksize = %d/%d" % (blksize, naux))

        p1 = 0
        for istep, chol in enumerate(df.loop(blksize=blksize)):
            p0, p1 = p1, p1 + chol.shape[0]
            jj[p0:p1] = numpy.dot(rho_tril, chol.T).T

        t0 = log.timer("THC", *t0)

        assert n < 20