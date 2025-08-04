!----------------------------------------------------------------------
!----------------------------------------------------------------------
!   Model from: Hu et al., 2018 (DOI: 10.1016/j.neuron.2018.02.024)
!   Modified, following Layer et al., 2021 (DOI: 10.3389/fncel.2021.754530)
!   Full system (with slow inactivation)
!----------------------------------------------------------------------
!----------------------------------------------------------------------

SUBROUTINE FUNC(NDIM, U, ICP, PAR, IJAC, F, DFDU, DFDP)
    !--------- ----

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: NDIM, IJAC, ICP(*)
    DOUBLE PRECISION, INTENT(IN) :: U(NDIM), PAR(*)
    DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
    DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM, *), DFDP(NDIM, *)

    DOUBLE PRECISION V, H, N, NBIS, SWT, SMUT
    DOUBLE PRECISION IAPP, GNAMUT, GNAWT, GK, GL, ENA, EK, EL, TEMPDIFF, SHIFTM, TEMP
    DOUBLE PRECISION GNATOT, PMUT
    DOUBLE PRECISION SINFWT, SINFMUT, TAUSWT, TAUSMUT, K, VH, SHIFTS
    DOUBLE PRECISION C, Q10M, Q10H, Q10N, Q10S, SHIFTV
    DOUBLE PRECISION INAWT, INAMUT, VSHIFTED, MINF, ALPHAM, BETAM, MINFMUT, ALPHAMMUT, BETAMMUT, ALPHAH, BETAH
    DOUBLE PRECISION IK, ALPHAN, BETAN, ALPHANBIS, BETANBIS
    DOUBLE PRECISION IL, QM, QH, QN, QS

    ! Define the state variables
    V = U(1)
    H = U(2)
    N = U(3)
    NBIS = U(4)
    SWT = U(5)
    SMUT = U(6)

    ! Define the system parameters
    IAPP = PAR(1)
    GNATOT = PAR(2)
    PMUT = PAR(3)
    GK = PAR(4)
    GL = PAR(5)
    TAUSMUT = PAR(6)
    EK = PAR(7)
    SHIFTV = PAR(8)
    TEMP = PAR(9)
    SHIFTS = PAR(10)

    SHIFTM = 0d0
    C = 0.9d0
    Q10M = 2.2d0
    Q10H = 2.9d0
    Q10S = 2.9d0
    Q10N = 3d0
    EL = -65d0
    TAUSWT = 30000d0
    ENA = 55d0
    K = -10d0
    VH = -60d0

    ! temperature
    TEMPDIFF = TEMP - 24d0
    QM = Q10M ** (TEMPDIFF/10)
    QH = Q10H ** (TEMPDIFF/10)
    QN = Q10N ** (TEMPDIFF/10)

    QS = Q10S ** ((TEMP-33d0)/10)

    ! sodium conductances
    GNAMUT = PMUT * GNATOT
    GNAWT = GNATOT * (1 - PMUT)

    ! sodium currents
    VSHIFTED = V - SHIFTV

    ! sodium activation wt
    ALPHAM = 0.2567 * (-(VSHIFTED + 60.84)) / (EXP(-(VSHIFTED + 60.84) / 9.722) - 1) * QM
    BETAM = 0.1133 * (VSHIFTED + 30.253) / (EXP((VSHIFTED + 30.253) / 2.848) - 1) * QM

    MINF = ALPHAM / (ALPHAM + BETAM)

    ! sodium fast inactivation wt
    ALPHAH = 0.00105 * EXP(-(VSHIFTED) / 20) * QH
    BETAH = 4.827 / (EXP(-(VSHIFTED + 18.646) / 12.452) + 1) * QH

    ! sodium slow inactivation
    SINFWT = 1 / (1 + EXP(-(V - VH) / K))
    SINFMUT = 1 / (1 + EXP(-(V - SHIFTS - VH) / K))

    INAWT = GNAWT * MINF ** 3 * H * SWT * (V - ENA)


    ! sodium activation mut
    ALPHAMMUT = 0.2567 * (-(VSHIFTED - SHIFTM + 60.84)) / (EXP(-(VSHIFTED - SHIFTM + 60.84) / 9.722) - 1) * QM
    BETAMMUT = 0.1133 * (VSHIFTED - SHIFTM + 30.253) / (EXP((VSHIFTED - SHIFTM + 30.253) / 2.848) - 1) * QM
    MINFMUT = ALPHAMMUT / (ALPHAMMUT + BETAMMUT)

    INAMUT = GNAMUT * MINFMUT ** 3 * H * SMUT * (V - ENA)

    ! potassium delayed-rectifier current
    IK = GK * N ** 3 * NBIS * (V - EK)

    ! potassium activation
    ALPHAN = 0.0610 * (-(V - 29.991)) / (EXP(-(v - 29.991) / 27.502) - 1) * QN
    BETAN = 0.001504 * (EXP(-V / 17.177)) * QN

    ALPHANBIS = 0.0993 * (-(V - 33.720)) / (EXP(-(v - 33.720) / 12.742) - 1) * QN
    BETANBIS = 0.1379 * (EXP(-V / 500)) * QN

    ! leak current
    IL = GL * (V - EL)

    ! Define the right-hand sides
    F(1) = 1 / C * (- INAWT - INAMUT - IK -IL + IAPP)
    F(2) = ALPHAH * (1 - H) - BETAH * H
    F(3) = ALPHAN * (1 - N) - BETAN * N
    F(4) = ALPHANBIS * (1 - NBIS) - BETANBIS * NBIS
    F(5) = (SINFWT - SWT) / TAUSWT * QS
    F(6) = (SINFMUT - SMUT) / TAUSMUT * QS


END SUBROUTINE FUNC
!----------------------------------------------------------------------

SUBROUTINE STPNT(NDIM, U, PAR, T)

    !----------------------------------------------------------------------

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: NDIM
    DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM), PAR(*)
    DOUBLE PRECISION, INTENT(IN) :: T

    PAR(1) = 0d0                       ! PARAMETER IAPP
    PAR(2) = 70d0                      ! PARAMETER GNATOT
    PAR(3) = 0d0                       ! PARAMETER PMUT
    PAR(4) = 15d0                      ! PARAMETER GK
    PAR(5) = 0.1d0                     ! PARAMETER GL
    PAR(6) = 30000d0                   ! PARAMETER TAUSMUT
    PAR(7) = -90d0                     ! PARAMETER EK
    PAR(8) = 20d0                      ! PARAMETER SHIFTV
    PAR(9) = 33d0                      ! PARAMETER TEMP
    PAR(10) = 0d0                      ! PARAMETER SHIFTS



END SUBROUTINE STPNT

SUBROUTINE PVLS
END SUBROUTINE PVLS

SUBROUTINE BCND(NDIM, PAR, ICP, NBC, U0, U1, FB, IJAC, DBC)
END SUBROUTINE BCND

SUBROUTINE ICND
END SUBROUTINE ICND

SUBROUTINE FOPT
END SUBROUTINE FOPT
!----------------------------------------------------------------------
