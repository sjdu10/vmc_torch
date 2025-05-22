
'''
PyTorch has its own implementation of backward function for SVD https://github.com/pytorch/pytorch/blob/291746f11047361100102577ce7d1cfa1833be50/tools/autograd/templates/Functions.cpp#L1577
We reimplement it with a safe inverse function (Lorentzian broadening) in case of degenerated singular values.
Theories and derivations of differentiable TN computation can be found in Liao et al. PHYSICAL REVIEW X 9, 031041 (2019).
'''

import numpy as np
import torch
import os, sys, itertools, time
import scipy.linalg
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

be_verbose = True
epsilon = 1e-12 
fix_sign = False

########## Utilities ##########

def safe_inverse(x):
    """ Lorentzian broadening of the inverse """
    return x/(x.pow(2) + epsilon)

def make_zeros(A):
    M,N = A.size()
    U = torch.eye(n=M,m=1)
    S = torch.zeros(1)
    Vh = torch.eye(n=1,m=N)
    return U,S,Vh

def is_one(T):
    return (torch.max(torch.abs(T-torch.ones_like(T))).detach().numpy() < epsilon)

########## SVD decomposition ##########

def SVDforward(A):
    M,N = A.size()
    if M * N == 0:
        raise ValueError(f"input matrix to custom SVD is size {(M,N)}")
    if not torch.all(torch.isfinite(A)): # A not finite
        raise ValueError("input matrix to custom SVD is not finite")
    try:
        U, S, Vh = torch.linalg.svd(A,full_matrices=False)
    except:
        if be_verbose:
            print('trouble in torch gesdd routine, falling back to gesvd')
        U, S, Vh = scipy.linalg.svd(A.detach().numpy(), full_matrices=False, lapack_driver='gesvd')
        U = torch.from_numpy(U)
        S = torch.from_numpy(S)
        Vh = torch.from_numpy(Vh)

    # if is_one(S): # A is isometry
    #     raise ValueError

    if fix_sign:
        # make SVD result sign-consistent across multiple runs
        for idx in range(U.size()[1]):
            if max(torch.max(U[:,idx]), torch.min(U[:,idx]), key=abs) < 0.0:
                U[:,idx] *= -1.0
                Vh[idx,:] *= -1.0
    return U,S,Vh

def SVDbackward(dU,dS,dVh,U,S,Vh):
    if not torch.all(torch.isfinite(dU)):
        raise ValueError("dU is not finite")
    if not torch.all(torch.isfinite(dS)):
        raise ValueError("dS is not finite")
    if not torch.all(torch.isfinite(dVh)):
        raise ValueError("dVh is not finite")
    M = U.size(0)
    N = Vh.size(1)

    F = (S - S[:, None])
    F = safe_inverse(F)
    F.diagonal().fill_(0)

    G = (S + S[:, None])
    G = safe_inverse(G)
    G.diagonal().fill_(0)

    UdU = U.t() @ dU
    VdV = Vh @ dVh.t()

    Su = (F+G)*(UdU-UdU.t())/2
    Sv = (F-G)*(VdV-VdV.t())/2

    dA = U @ (Su + Sv + torch.diag(dS)) @ Vh
    Su = Sv = UdU = VdV = G = F = None   # help with memory
    Sinv = safe_inverse(S)
    NS = S.size(0)
    Sinv = Sinv.reshape((1,NS))
    if (M>NS):
        ##dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt
        #dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU*safe_inverse(S)) @ Vt
        # the following is a rewrite of the above one-liner to reduce memory
        tmp1 = (dU*Sinv) @ Vh
        tmp2 = U.t() @ tmp1
        tmp2 = U @ tmp2
        dA += (tmp1 - tmp2)
    if (N>NS):
        ##dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        #dA = dA + (U*safe_inverse(S)) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        # the following is a rewrite of the above one-liner to reduce memory
        tmp1 = (U*Sinv) @ dVh
        tmp2 = tmp1 @ Vh.t()
        tmp2 = tmp2 @ Vh
        dA += (tmp1 - tmp2)
    if not torch.all(torch.isfinite(dA)):
        raise ValueError("dA is not finite")
    return dA

class SVD(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        U,S,Vh = SVDforward(A)
        self.save_for_backward(U, S, Vh)
        return U, S, Vh
    @staticmethod
    def backward(self, dU, dS, dVh):
        U, S, Vh = self.saved_tensors
        return SVDbackward(dU,dS,dVh,U,S,Vh)
    


########## QR decomposition ##########

def copyltu(A):
    tril0 = A.tril(diagonal=0)
    tril1 = A.tril(diagonal=-1)
    return tril0 + tril1.t()
def QRbackward_deep(Q,R,dQ,dR):
    M = R@dR.t() - dQ.t()@Q
    M = copyltu(M)        
    tmp = dQ + Q@M
    dA = torch.linalg.solve_triangular(R,tmp.t(),left=True,upper=True)
    if not torch.all(torch.isfinite(dA)):
        raise ValueError("dA is not finite")
    return dA.t() 
def QRbackward_wide(A,Q,R,dQ,dR):
    M,N = A.size()
    X,Y = A.split((M,N-M),dim=1)
    U,V = R.split((M,N-M),dim=1)
    dU,dV = dR.split((M,N-M),dim=1)

    tmp = dQ+Y@dV.t()
    M = U@dU.t() - tmp.t()@Q
    M = copyltu(M)
    tmp = tmp + Q @ M 
    dX = torch.linalg.solve_triangular(U,tmp.t(),left=True,upper=True)
    if not torch.all(torch.isfinite(dX)):
        raise ValueError("dX is not finite")
    return torch.cat((dX.t(),Q@dV),dim=1)

class QR(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        M,N = A.size()
        if M * N == 0:
            raise ValueError(f"input matrix to custom QR is size {(M,N)}")
        if not torch.all(torch.isfinite(A)): # A not finite
            print(A)
            raise ValueError("input matrix to custom QR is not finite")
        try:
            Q, R = torch.linalg.qr(A,mode='reduced')
        except:
            if be_verbose:
                print('trouble in torch gesdd routine, falling back to scipy')
            Q, R = scipy.linalg.svd(A.detach().numpy(), mode='economic')
            Q = torch.from_numpy(Q)
            R = torch.from_numpy(R)

        diag = R.diag()
        # if is_one(diag):
        #     print(R)
        #     raise ValueError
        # inds = torch.abs(diag) < epsilon
        # if len(inds) > 0: # rank deficient, revert to svd
        #     U,S,Vh = SVDforward(A)
        #     SVh = S.reshape((S.size(0),1)) * Vh
        #     self.save_for_backward(U, S, Vh)
        #     return U, SVh

        if fix_sign:
            sign = torch.sign(diag).reshape((1,-1))
            Q = Q * sign
            R = R * sign.t()
        self.save_for_backward(A,Q,R)
        return Q,R
        
    @staticmethod
    def backward(self, dQ, dR):
        M1,M2,M3 = self.saved_tensors
        if len(M2.size())==1: # rank-deficient, do svd
            U,S,Vh = M1,M2,M3
            dU,dSVh = dQ,dR
            dS = torch.diag(dSVh @ Vh.t())
            dVh = S.reshape((S.size(0),1)) * dSVh
            return SVDbackward(dU,dS,dVh,U,S,Vh)
        if not torch.all(torch.isfinite(dQ)):
            raise ValueError("dQ is not finite")
        if not torch.all(torch.isfinite(dR)):
            raise ValueError("dR is not finite")
        A,Q,R = M1,M2,M3
        M,N = A.size()
        if M>=N:
            return QRbackward_deep(Q,R,dQ,dR)
        else:
            return QRbackward_wide(A,Q,R,dQ,dR)
        

########## Test ##########


def test_svd():
    M, N = 50, 20
    torch.manual_seed(2)
    input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    t0 = time.time()
    assert(torch.autograd.gradcheck(SVD.apply, (input), eps=1e-6, atol=1e-4))
    print(f"SVD Test Pass for {M},{N}! time={time.time()-t0}")

    M, N = 20, 50
    torch.manual_seed(2)
    input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    t0 = time.time()
    assert(torch.autograd.gradcheck(SVD.apply, (input), eps=1e-6, atol=1e-4))
    print(f"SVD Test Pass for {M},{N}! time={time.time()-t0} ")

    M, N = 20, 20
    torch.manual_seed(2)
    input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    t0 = time.time()
    assert(torch.autograd.gradcheck(SVD.apply, (input), eps=1e-6, atol=1e-4))
    print(f"SVD Test Pass for {M},{N}! time={time.time()-t0}")
        
def test_qr():
    M, N = 50, 20
    torch.manual_seed(2)
    input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    t0 = time.time()
    assert(torch.autograd.gradcheck(QR.apply, (input), eps=1e-6, atol=1e-4))
    print(f"QR Test Pass for {M},{N}! time={time.time()-t0}")

    M, N = 20, 50
    torch.manual_seed(2)
    input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    t0 = time.time()
    assert(torch.autograd.gradcheck(QR.apply, (input), eps=1e-6, atol=1e-4))
    print(f"QR Test Pass for {M},{N}! time={time.time()-t0}")

    M, N = 20, 20
    torch.manual_seed(2)
    input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    t0 = time.time()
    assert(torch.autograd.gradcheck(QR.apply, (input), eps=1e-6, atol=1e-4))
    print(f"QR Test Pass for {M},{N}! time={time.time()-t0}")

if __name__=='__main__':
    test_svd()
    test_qr()



