import subprocess


'''
 Method 2: compute HVSR curve based on the cope by O'Neill and OpenHVSR
'''
# HVSR analysis and inversion for subsurface structure
# Developed by Craig O'Neill 2023
# Distributed under the accompanying MIT licence
# Use at own discretion

import numpy as np
import numpy as _np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft, fft, ifft, fftfreq
from numba import jit
from scipy.optimize import minimize
from scipy.signal import periodogram, welch,lombscargle
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import griddata, Rbf
from scipy.optimize import optimize
import glob
import copy

class HVSRForwardModels(object):
    def __init__(self,
				 fre1 = 1,
				 fre2 = 60,
				 Ds = np.array( [0.05, 0.01, 0.01,0.01]),
                 Dp = np.array( [0.05, 0.01, 0.01,0.01]),
				 h = np.array([10, 200,100, 1e3]),
				 ro = np.array([1000, 2000, 2000, 2000])/1000,
				 Vs = np.array([500, 1500, 1500, 1500]),
				 Vp = np.array([500, 3000, 3000, 3000]),
                 ex = 0.0,
                 fref = 1.0,
                 f = np.logspace(np.log10(1),np.log10(60),500),
				 filename = None
					):
            self.fre1 = fre1
            self.fre2 = fre2
            self.Ds = Ds
            self.Dp = Dp
            self.h = h
            self.ro = ro
            self.Vs = Vs
            self.Vp = Vp
            self.ex = ex
            self.fref=fref
            self.f = f


    def Transfer_Function(self,):
        """

        Adopted from Transfer Function Tool
        Weaponised by Craig O'Neill for inversion of HVSR
        Assumes the transfer function approximates HVSR response (which it turns out it not bad, see Nakamura).
        Note uses damping D, not Qs.

        Compute the SH-wave transfer function using Knopoff formalism
        (implicit layer matrix scheme). Calculation can be done for an
        arbitrary angle of incidence (0-90), with or without anelastic
        attenuation (qs is optional).

        It return the displacements computed at arbitrary depth.
        If depth = -1, calculation is done at each layer interface
        of the profile.

        NOTE: the implicit scheme is simple to understand and to implement,
        but is also computationally intensive and memory consuming.
        For the future, an explicit (recursive) scheme should be implemented.

        :param float or numpy.array freq:
        array of frequencies in Hz for the calculation

        :param numpy.array hl:
            array of layer's thicknesses in meters (half-space is 0.)

        :param numpy.array vs:
            array of layer's shear-wave velocities in m/s

        :param numpy.array dn:
            array of layer's densities in kg/m3

        :param numpy.array qs:
            array of layer's shear-wave quality factors (adimensional)

        :param float inc_ang:
            angle of incidence in degrees, relative to the vertical
            (default is vertical incidence)

        :param float or numpy.array depth:
            depths in meters at which displacements are calculated
            (default is the free surface)

        :return numpy.array dis_mat:
            matrix of displacements computed at each depth (complex)
        """

        #freq=np.linspace(self.fre1,self.fre2,500)
        #print("Initial freq!!",freq)
        hl = self.h
        vs = self.Vs
        dn = self.ro
        Ds = self.Ds
        freq=self.f
        qs = np.ones_like(vs)*(1.0/(2.0*Ds))
        inc_ang=0.
        depth=0.
        #print("Transfer function: Vs:",vs)

        # Precision of the complex type
        CTP = 'complex128'

        # Check for single frequency value
        if isinstance(freq, (int, float)):
            freq = _np.array([freq])

        # Model size
        lnum = len(hl)
        fnum = len(freq)

        # Variable recasting to numpy complex
        hl = _np.array(hl, dtype=CTP)
        vs = _np.array(vs, dtype=CTP)
        dn = _np.array(dn, dtype=CTP)

        # Attenuation using complex velocities
        if qs is not None:
            qs = _np.array(qs, dtype=CTP)
            vs *= ((2.*qs*1j)/(2.*qs*1j-1.))

        # Conversion to angular frequency
        angf = 2.*_np.pi*freq

        # Layer boundary depth (including free surface)
        #print("hl",hl,hl.dtype)
        bounds = self.interface_depth(self)

        # Check for depth to calculate displacements
        if isinstance(depth, (int, float)):
            if depth < 0.:
                depth = _np.array(bounds)
            else:
                depth = _np.array([depth])
        znum = len(depth)

        # -------------------------------------------------------------------------
        # Computing angle of propagation within layers

        iD = _np.zeros(lnum, dtype=CTP)
        iM = _np.zeros((lnum, lnum), dtype=CTP)

        iD[0] = _np.sin(inc_ang)
        iM[0, -1] = 1.

        for nl in range(lnum-1):
            iM[nl+1, nl] = 1./vs[nl]
            iM[nl+1, nl+1] = -1./vs[nl+1]

        iA = _np.linalg.solve(iM, iD)
        iS = _np.arcsin(iA)

        # -------------------------------------------------------------------------
        # Elastic parameters

        # Lame Parameters : shear modulus
        mu = dn*(vs**2.)

        # Horizontal slowness
        ns = _np.cos(iS)/vs

        # -------------------------------------------------------------------------
        # Data vector initialisation

        # Layer's amplitude vector (incognita term)
        amp_vec = _np.zeros(lnum*2, dtype=CTP)

        # Layer matrix
        lay_mat = _np.zeros((lnum*2, lnum*2), dtype=CTP)

        # Input motion vector (known term)
        inp_vec = _np.zeros(lnum*2, dtype=CTP)
        inp_vec[-1] = 1.

        # Output layer's displacement matrix
        dis_mat = _np.zeros((znum, fnum), dtype=CTP)

        # -------------------------------------------------------------------------
        # Loop over frequencies

        for nf in range(fnum):

            # Reinitialise the layer matrix
            lay_mat *= 0.

            # Free surface constraints
            lay_mat[0, 0] = 1.
            lay_mat[0, 1] = -1.

            # Interface constraints
            for nl in range(lnum-1):
                row = (nl*2)+1
                col = nl*2

                exp_dsa = _np.exp(1j*angf[nf]*ns[nl]*hl[nl])
                exp_usa = _np.exp(-1j*angf[nf]*ns[nl]*hl[nl])

                # Displacement continuity conditions
                lay_mat[row, col+0] = exp_dsa
                lay_mat[row, col+1] = exp_usa
                lay_mat[row, col+2] = -1.
                lay_mat[row, col+3] = -1.

                # Stress continuity conditions
                lay_mat[row+1, col+0] = mu[nl]*ns[nl]*exp_dsa
                lay_mat[row+1, col+1] = -mu[nl]*ns[nl]*exp_usa
                lay_mat[row+1, col+2] = -mu[nl+1]*ns[nl+1]
                lay_mat[row+1, col+3] = mu[nl+1]*ns[nl+1]

            # Input motion constraints
            lay_mat[-1, -1] = 1.

            # Solving linear system of wave's amplitudes
            try:
                amp_vec = _np.linalg.solve(lay_mat, inp_vec)
            except:
                amp_vec[:] = _np.nan

            # ---------------------------------------------------------------------
            # Solving displacements at depth

            for nz in range(znum):

                # Check in which layer falls the calculation depth
                if depth[nz] <= hl[0]:
                    nl = 0
                    dh = depth[nz]
                elif depth[nz] > bounds[-1]:
                    nl = lnum-1
                    dh = depth[nz] - bounds[-1]
                else:
                    # There might be a more python way to do that...
                    nl = map(lambda x: x >= depth[nz], bounds).index(True) - 1
                    dh = depth[nz] - bounds[nl]

                # Displacement of the up-going and down-going waves
                exp_dsa = _np.exp(1j*angf[nf]*ns[nl]*dh)
                exp_usa = _np.exp(-1j*angf[nf]*ns[nl]*dh)

                dis_dsa = amp_vec[nl*2]*exp_dsa
                dis_usa = amp_vec[nl*2+1]*exp_usa

                dis_mat[nz, nf] = dis_dsa + dis_usa

        return freq, np.abs(dis_mat[0,:])


# =============================================================================

    def interface_depth(self, dtype='complex128'):
        """
        Utility to calcualte the depth of the layer's interface
        (including the free surface) from a 1d thickness profile.

        :param numpy.array hl:
            array of layer's thicknesses in meters (half-space is 0.)

        :param string dtype:
            data type for variable casting (optional)

        :return numpy.array depth:
            array of interface depths in meters
        """
        CTP = 'complex128'
        hl2 = self.h
        hl = _np.array(hl2, dtype=CTP)
        #print("In interface, hl:",hl2)
        depth = np.array([sum(hl[:i]) for i in range(len(hl))])
        depth = _np.array(depth.real, dtype="float64")

        return depth


    def HV(self,):

        s_amp=self.HV3(self.Vs, self.ro, self.h, self.Ds, self.ex, self.fref, self.f)
        p_amp=self.HV3(self.Vp, self.ro, self.h, self.Dp, self.ex, self.fref, self.f)
        hvsr = s_amp/p_amp
        return(self.f,hvsr)

    def HV3(self,c, ro, h, d, ex, fref, f):
        # Code adopted From Albarello el al. Suppl. Mat/BW, following from Model HVSR (Herak) and OpenHVSR.
        # Migrated to pure python and benchmarked by Craig O'Neill Nov 2023.
        q = 1/(2*d)
        ns = len(c)
        nf = len(f)
        TR=np.zeros((50,1),dtype=complex);
        AR=np.zeros((50,1),dtype=complex);
        qf=np.zeros((ns,nf),dtype=complex);
        T=np.zeros((ns,1),dtype=complex);
        A=np.zeros((ns,1),dtype=complex);
        FI=np.zeros((ns,1),dtype=complex);
        Z=np.zeros((ns,1),dtype=complex);
        X=np.zeros((ns,1),dtype=complex);
        FAC=np.zeros((ns,nf),dtype=complex);
        frref = fref
        frkv = f
        qf = np.zeros((ns, nf))
        #print('size of q:',q.size)
        #print('size of frkv:',frkv.size)
        #print('ns and nf:',ns,nf)
        for j in range(ns):
            for i in range(nf):
                #print(j,i)
                qf[j, i] = q[j] * frkv[i] ** ex

        idisp = 0
        if frref > 0:
            idisp = 1

        TR = np.zeros(ns - 1)
        AR = np.zeros(ns - 1)

        for I in range(ns - 1):
            TR[I] = h[I] / c[I]
            AR[I] = ro[I] * c[I] / ro[I + 1] / c[I + 1]

        NSL = ns - 1
        TOTT = sum(TR)

        #X = np.zeros(NSL + 1, dtype=np.complex128)
        #Z = np.zeros(NSL + 1, dtype=np.complex128)
        X[0] = 1.0+0.0j
        Z[0] = 1.0 + 0.0j
        II = 1j

        korak = 1
        if idisp == 0:
            FJM1 = 1
            FJ = 1

        FAC = np.zeros((NSL + 2, nf), dtype=np.complex128)

        for J in range(1, NSL + 1):
            for ii in range(nf):
                FAC[J - 1, ii] = 2 / (1 + np.sqrt(1 + qf[J - 1, ii] ** (-2))) * (1 - 1j / qf[J - 1, ii])
                FAC[J - 1, ii] = np.sqrt(FAC[J - 1, ii])

        FAC[NSL, :nf] = 1
        qf[NSL, :nf] = 999999

        jpi = 1 / 3.14159

        AMP = np.zeros(nf)

        for k in range(0, nf, korak):
            ALGF = np.log(frkv[k] / frref)

            for J in range(2, NSL + 2):
                if idisp != 0:
                    FJM1 = 1 + jpi / qf[J - 2, k] * ALGF
                    FJ = 1 + jpi / qf[J - 1, k] * ALGF

                T[J - 2] = TR[J - 2] * FAC[J - 2, k] / FJM1
                A[J - 2] = AR[J - 2] * FAC[J - 1, k] / FAC[J - 2, k] * FJM1 / FJ
                FI[J - 2] = 6.283186 * frkv[k] * T[J - 2]
                ARG = 1j * FI[J - 2]

                CFI1 = np.exp(ARG)
                CFI2 = np.exp(-ARG)

                Z[J - 1] = (1 + A[J - 2]) * CFI1 * Z[J - 2] + (1 - A[J - 2]) * CFI2 * X[J - 2]
                Z[J - 1] = Z[J - 1] * 0.5

                X[J - 1] = (1 - A[J - 2]) * CFI1 * Z[J - 2] + (1 + A[J - 2]) * CFI2 * X[J - 2]
                X[J - 1] = X[J - 1] * 0.5

            AMP[k] = 1 / abs(Z[NSL])

        return AMP


